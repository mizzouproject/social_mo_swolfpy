# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:22:00 2023

@author: jepiguti
"""
from .Optimization import Optimization
from .LCA_matrix import LCA_matrix
import numpy as np
import pandas as pd
import json
import math
#pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as MultiObjMinimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.visualization.scatter import Scatter
from pymoo.core.repair import Repair
from pymoo.core.result import Result
from pymoo.core.callback import Callback
from pymoo.termination.ftol import calc_delta_norm
from pymoo.util.normalization import normalize
from pymoo.indicators.igd import IGD

import shutup; shutup.please()
from datetime import datetime

import matplotlib.pyplot as plt
import oapackage
from oapackage.oahelper import create_pareto_element
from sklearn import preprocessing

import json
from json import JSONEncoder

from .Topsis import Topsis

class MultiOptimization():
    def __init__(self, functional_unit, method, project, bw2_fu):
        """Constructor of the class

        Args:
            functional_unit (dict): e.g. {('waste', scenary_name): 1}
            method (list): e.g. [{'name': 'Social Cost', 'method': ('SwolfPy_Social','SwolfPy'), 'unit': '$/Mg'}, {}...]
            project (swolfpy.Project.Project): Project
            bw2_fu (bw2data.backends.peewee.proxies.Activity): Brightway functional unit
        """
        self.tp = Topsis()
        self.optimization_list = []     # list of Optimization objects (Optimization.py)
        self.configArray = []           # list to setup each Optimization object
        self.objs = []                  # list of objective functions
        self.constr_eq = []             # list of equality constrains
        self.constr_ieq = []            # list of inequality constrains
        self.cons = []                  # list of equality and inequality constrains
        self.n_var = 0                  # number of project.parameters_list + n_scheme_vars of Optimization object
        for m in method:
            mth = []
            mth.append(m['method'])
            tmpOpt = Optimization(functional_unit=functional_unit,method=mth,project=project)
            self.objs.append(tmpOpt._objective_function)
            self.optimization_list.append(tmpOpt)
        self.method_list = method       # list of methods
        self.res = []                   # to save the result of the algorithm
        self.bw2_fu = bw2_fu
        self.has_result = False         # boolean to know if the algorithm has a result (res.X exists)
        self.results = []               # to save the DataFrame of the function report_res()
        self.has_history = False        # boolean to validate if the history of the algorithm was saved
        self.running_data = []          # list to save the data of the RunningMetric class
        self.running_history = []       # list to save the history of the RunningMetric class
        self.igd = []                   # list to save igd from running_data variable
        self.imported_from_json = False # boolean to validate if a json was imported from a previous run of the optimization
        self.history_data = dict()      # self.get_history() of a previous run of the optimization when uploaded from a json
        self.result_topsis = []         # save the output of function get_topsis
        self.individual_topsis = []     # indexes of the individuals from results
    
    def config(self, project):
        """Iterate each Optimization object and call config function of each one

        Args:
            project (swolfpy.Project.Project): Project
        """
        for optElem in self.optimization_list:
            self.configArray = optElem.config(project)
    
    def set_config(self):
        """Call set_config function of each Optimization list and count the total number of vars
        """
        for optElem in self.optimization_list:
            optElem.set_config(self.configArray)            
        self.n_var = len(self.optimization_list[0].project.parameters_list) + self.optimization_list[0].n_scheme_vars

    def set_constraints(self, constraints=None, collection=False):
        """Set inequality constrains to each Optimization object

        Args:
            constraints (dict, optional): Dictionary with inequality constraints e.g. {'limit':154529, 'KeyType':'Process','ConstType':"<="}. Defaults to None.
            collection (bool, optional): To validate if collection exists. Defaults to False.
        """
        for optElem in self.optimization_list:
            optElem.oldx = [0 for i in range(self.n_var)]
            optElem.magnitude = len(str(int(abs(optElem.score))))
            optElem.constraints = constraints
            optElem.collection = collection
            optElem.cons = optElem._create_constraints(inverse=-1)
        self.cons = self.optimization_list[0].cons

    def multi_obj_optimization(self, constraints=None, collection=False, pop_size=30,
                                 n_offsprings=None, eliminate_duplicates=True, 
                                 termination=40, seed=1, verbose=True, save_history=False,
                                 repair=False, verbose_repair=False):
        """Performs multi-objective optimization using NSGA-II algorithm

        Args:
            constraints (dict, optional): Dictionary with inequality constraints e.g. {'limit':154529, 'KeyType':'Process','ConstType':"<="}. Defaults to None.
            collection (bool, optional): To validate if collection exists. Defaults to False.
            pop_size (int, optional): Population size. Defaults to 30.
            n_offsprings (int, optional): Number of offsprings. Defaults to None.
            eliminate_duplicates (bool, optional): Whether or not the algorithm eliminate duplicates in results. Defaults to True.
            termination (int, optional): Number of generations when the algorithm will stop. Defaults to 40.
            seed (int|bool, optional): To guarantee reproducibility of results. True generates random results everytime, while a int number will gives the same result everytime. Defaults to 1.
            verbose (bool, optional): To print results in each generation. Defaults to True.
            save_history (bool, optional): Save history of the algorithm to later generate running metric analysis. Defaults to False.
            repair (bool, optional): Whether or not use a function to guarantee a faster feasibility of equality constraints. Defaults to False.
            verbose_repair (bool, optional): To print the results of repair function in each generation. Defaults to False.

        Returns:
            pymoo.core.result.py: Object that contains algorithm results, None when there is not result
        """        
        number_of_results = 0
        self.results = []
        self.has_result = False
        self.has_history = False
        self.imported_from_json = False
        self.running_data = []
        self.running_history = []
        """
        Initialization of each Optimization object inside the optimization_list
        """   
        if len(self.cons) == 0 or constraints != None or collection != False:
            self.set_constraints(constraints, collection)

        for constraint in self.cons:
            if constraint['type'] == 'eq':
                self.constr_eq.append(constraint['fun'])
            if constraint['type'] == 'ineq':
                self.constr_ieq.append(constraint['fun'])
        """
        Setup of the Functional Problem
        n_var       -> number of variables
        objs        -> objective functions
        constr_ieq  -> inequality constraints
        constr_eq   -> equality constraints
        xl          -> lower limit of decision variables
        xu          -> upper limit of decision variables
        """
        xl = [0 for _ in range(self.n_var)]
        xu = [1 for _ in range(self.n_var)]
        problem = FunctionalProblem(n_var=self.n_var,
                                    objs=self.objs,
                                    constr_ieq=self.constr_ieq,
                                    constr_eq=self.constr_eq,
                                    xl=xl,
                                    xu=xu)
        print("Problem object created")

        repairObject = None
        if repair:
            equality_var_array = dict()        
            for key in self.optimization_list[0].project.parameters.param_uncertainty_dict.keys():
                equality_var_array[key] = len(self.optimization_list[0].project.parameters.param_uncertainty_dict[key])
            repairObject = FractionSumOneRepair(equality_var_array=equality_var_array, verbose=verbose_repair)
            print("Repair object created")

        """
        Setup of the Algorithm
        """  
        algorithm = NSGA2(pop_size=pop_size,
                    sampling=LHS(),
                    crossover=SBX(prob=0.9), #eta=15, prob=0.9
                    mutation=PolynomialMutation(prob=0.03), #prob=0.9, eta=20
                    n_offsprings=n_offsprings,
                    eliminate_duplicates=eliminate_duplicates,
                    repair=repairObject)
        print("Algorithm object created")
        """
        Minimize
        """  
        start_time = datetime.now() # current date and time
        start_date_time = start_time.strftime("%H:%M:%S")
        print("Optimization started at: "+start_date_time)
        res = MultiObjMinimize(problem=problem,
                                algorithm=algorithm,
                                termination=('n_gen', termination),
                                seed=seed,
                                verbose=verbose,
                                callback=HistoryCallback() if save_history else DummyCallback(),
                                return_least_infeasible=True)
        end_time = datetime.now() # current date and time
        end_date_time = end_time.strftime("%H:%M:%S")
        print("Optimization finished at: "+end_date_time)
        total_time = datetime.strptime(end_date_time, "%H:%M:%S") - datetime.strptime(start_date_time, "%H:%M:%S")
        print(f"Optimization lasted: {total_time.total_seconds()} seconds")
        try:
            number_of_results = len(res.X)
            self.has_result = True
            self.has_history = save_history
            if save_history:
                self.running_data = res.algorithm.callback.running_data
                self.running_history = res.algorithm.callback.running_history
                self.history_data = res.algorithm.callback.history_data
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('Resulset Size = '+str(number_of_results))
            self.res = res
        return res
    
    def report_res(self, Opt=False, fileName='', update=False):
        """DataFrame with values for objective functions and decision variables for each generation from the pareto front
        
        Args:
            Opt (bool, optional): Whether or not an optimization object is given. Defaults to False.
            fileName (str, optional): Name of the file in which the report will be saved in csv format. Defaults to ''.
            update (bool, optional): To update the report when an optimization object is given. Defaults to False.
        """
        if self.has_result == False:
            print("NO RESULTSET TO PROCESS A REPORT")
        else:
            default_cost_values = [('SwolfPy_Operational_Cost', 'SwolfPy'), ('SwolfPy_Capital_Cost', 'SwolfPy'), ('SwolfPy_Total_Cost', 'SwolfPy'), ('SwolfPy_Social', 'SwolfPy')]
            used_cost = []
            dict_method = []
            if Opt == False:
                Opt = self.optimization_list[0]
            
            if len(self.results) == 0 or update:
                results = pd.DataFrame()
                processes = ['Collection','LF','WTE','AD','Composting','SS_MRF','Reprocessing']
                for individual in range(len(self.res.X)):
                    Opt._objective_function(self.res.X[individual])            
                    individual_name = "ind_"+str(individual+1)
                    if 'unit' in results.columns:
                        results.insert(len(results.columns),individual_name,np.nan)
                    else:
                        index = ['Diversion']
                        for m in self.method_list:
                            dict_method.append({'name':m['name'], 'method':m['method'], 'unit':m['unit']})
                            used_cost.append(m['method'])
                            index.append(m['name'])
                        
                        for dc in default_cost_values:
                            if dc not in used_cost:
                                used_cost.append(dc)
                                name = dc[0].replace('SwolfPy_','').replace('_',' ') # Social 2
                                if name == 'Social':
                                    # name += ' Cost'
                                    name += ' Management Cost'
                                elif name == 'Social 2':
                                    # name += ' Cost'
                                    name = 'Social Community Cost'
                                dict_method.append({'name':name, 'method':dc, 'unit': '$/Mg'})
                                index.append(name)
                                
                        index = index + processes + ['IND']
                        results = pd.DataFrame(columns=['unit'],
                                        index=index)
                        
                        for dm in dict_method:
                            results.at[dm['name'], 'unit'] = dm['unit']
                            
                        results.at['Diversion', 'unit'] = '%'
                        for process in processes:
                            results.at[process, 'unit'] = 'Mg/yr'
                    
                    results.at['IND', individual_name] = str(individual+1)
                    
                    for dm in dict_method:
                        Opt.switch_method(dm['method'])
                        Opt.lcia()
                        results.at[dm['name'], individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)
        
                    for process in processes:
                        results.at[process, individual_name] = round(LCA_matrix.get_mass_flow(Opt, process))
        
                    results.at['Diversion', individual_name] = round((1-results[individual_name]['LF']/results[individual_name]['Collection'])*100,2)
                self.results = results
            else:
                results = self.results
        
            if len(fileName) > 0:
                results.to_csv(fileName+'.csv')
        return(results)

    def test_pareto(self, value=None, show=False):
        """Test set of results from the algorithm to confirm that each value belongs to the Pareto front
        Library oapackage is used

        Args:
            value (_type_, optional): Set of results to be tested. None uses the results from the algorithm. Defaults to None.
            show (bool, optional): Print whether or not results belong to the pareto front. Defaults to False.

        Returns:
            oalib.ParetoMultiDoubleLong: Object ParetoMultiDoubleLong
        """
        if value != None:
            datapoints = value
        else:
            if self.has_result:
                datapoints = self.res.F.T
            else:
                print('No result to work with')
                return
            
        pareto = oapackage.ParetoMultiDoubleLong()

        for ii in range(0, datapoints.shape[1]):
            values = []
            for jj in range(0, datapoints.shape[0]):
                values.append([datapoints[jj,ii]]) 
            val = create_pareto_element(values, pareto=pareto)
            pareto.addvalue(val, ii)

        if show:
            pareto.show(verbose=1)
        return pareto

    def dynamic_scatter_plot_2D(self, x_name, y1_name, y2_name):
        """Scatter plot for three objective functions in 2D

        Args:
            x_name (string): Objective function to plot in x axis
            y1_name (string): Objective function to plot on left y axis
            y2_name (string): Objective function to plot on right y axis
        """
        results = self.report_res()
        data_to_plot = results.T.drop(index='unit')
        names = data_to_plot.columns
        error = ''
        if x_name not in names:
            error += 'Name: ' + x_name + ' for x_name is not correct. '
        if y1_name not in names:
            error += 'Name: ' + y1_name + ' for y1_name is not correct. '
        if y2_name not in names:
            error += 'Name: ' + y2_name + ' for y2_name is not correct. '
        if len(error) > 0:
            print(error)
            return
        
        fig, ax1 = plt.subplots()

        x = data_to_plot[x_name].values
        y1 = data_to_plot[y1_name].values
        y2 = data_to_plot[y2_name].values
        
        ax2 = ax1.twinx()
        ax1.scatter(x, y1, c='g')
        ax2.scatter(x, y2, c='b')
        
        x_label = results['unit'][x_name]
        y1_label = results['unit'][y1_name]
        y2_label = results['unit'][y2_name]

        ax1.set_xlabel(x_name + ' (' + x_label + ')')
        ax1.set_ylabel(y1_name + ' (' + y1_label + ')', color='g')
        ax2.set_ylabel(y2_name + ' (' + y2_label + ')', color='b')
        
        # Display the plot
        plt.show()

    def dynamic_scatter_plot_3D(self, x_name, y_name, z_name):
        """Scatter plot for three objective functions in 3D

        Args:
            x_name (string): Objective function to plot in x axis
            y_name (string): Objective function to plot in y axis
            z_name (string): Objective function to plot in z axis
        """
        results = self.report_res()
        data_to_plot = results.T.drop(index='unit')
        names = data_to_plot.columns
        error = ''
        if x_name not in names:
            error += 'Name: ' + x_name + ' for x_name is not correct. '
        if y_name not in names:
            error += 'Name: ' + y_name + ' for y_name is not correct. '
        if z_name not in names:
            error += 'Name: ' + z_name + ' for z_name is not correct. '
        if len(error) > 0:
            print(error)
            return
        base_data = data_to_plot[[x_name, y_name, z_name]]
        normalized_base_data = preprocessing.normalize(base_data)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')

        # Data for three-dimensional scattered points
        zdata = normalized_base_data.T[2]
        xdata = normalized_base_data.T[0]
        ydata = normalized_base_data.T[1]
        
        ax.scatter3D(xdata, ydata, zdata);
    
        ax.set_xlabel(x_name, fontsize=10, labelpad=13)
        ax.set_ylabel(y_name, fontsize=10, labelpad=13)
        ax.set_zlabel(z_name, fontsize=10, labelpad=3, rotation= 'vertical')

        plt.show()
    
    def dynamic_multi_scatter_plot_2D(self, column_names):
        """Scatter plot for two objective functions at a time in 2D

        Args:
            column_names (list): e.g. [('Total cost','Social cost'),('Total cost', 'GWP')]
        """
        results = self.report_res()
        data_to_plot = results.T.drop(index='unit')
        names = data_to_plot.columns
        error = ''
        for c in column_names:
            if c[0] not in names:
                error += 'Name: ' + c[0] + ' doesn\'t exist in the report. '
            if c[1] not in names:
                error += 'Name: ' + c[1] + ' doesn\'t exist in the report. '
        if len(error) > 0:
            print(error)
            return

        figure, ax1 = plt.subplots(len(column_names), figsize=(6,len(column_names)*4))

        i = 0
        for c in column_names:
            if len(column_names) == 1:
                ax = ax1
            else:
                ax = ax1[i]  
            ax.scatter(data_to_plot[c[0]], data_to_plot[c[1]])
            ax.set_title(c[0] + ' vs ' + c[1])
            x_label = c[0] + ' (' + results['unit'][c[0]] + ')'
            y_label = c[1] + ' (' + results['unit'][c[1]] + ')'
            ax.set(xlabel=x_label, ylabel=y_label)
            i += 1
            
        figure.tight_layout()
        
        # Display the plot
        plt.show()

    def dynamic_bar_chart_tech_mix(self, current=None, column_names=None, individual=None, current_diversion=None, show_diversion=False):
        """Bar Chart for individuals. Each bar containts the distribution of waste in percentage treated by each process model (e.g. LF)

        Args:
            current (list, optional): List of values that correspond to the current situation. This is represented in the first bar of the chart. Defaults to None. E.g. {'LF':220773, 'WTE':0, 'AD':0, 'Composting':9104}
            column_names (list, optional): List of process models to be considered for the chart. Defaults to None. E.g. ['LF', 'WTE', 'AD', 'Composting', 'SS_MRF', 'Reprocessing']
            individual (list, optional): List of individuals to be included in the chart. Defaults to None.
            current_diversion (float, optional): Value that corresponds to the current diversion. Defaults to None.
            show_diversion (bool, optional): Whether or not diversion data is included in the chart. Defaults to False.
        """
        colors = {0:'gold',1:'darkorange',2:'blue',3:'limegreen',4:'red',5:'violet'}
        default_column_names = ['LF', 'WTE', 'AD', 'Composting', 'SS_MRF', 'Reprocessing']
        width = 0.35
        results = self.report_res()
        data_to_plot = results.T.drop(index='unit')
        if show_diversion and current != None:
            if current_diversion == None:
                print("Not a value for current diversion.")
                return
            elif current_diversion < 1 and current_diversion > 100:
                print("Current diversion should be a value between 1 and 100.")
                return
        
        if column_names == None:
            column_names = default_column_names
        else:
            error_column_name = []
            for cn in column_names:
                if cn not in default_column_names:
                    error_column_name.append(cn)
            if len(error_column_name) > 0:
                print("Column name incorrect: "+str(error_column_name))
                return
        if current != None:
            if len(current) != len(column_names):
                print("Current variable has lenght of "+str(len(current))+", expected "+str(len(column_names)))
                return
        error_individual = []
        individuals = None
        if individual != None:
            for ind in individual:
                if ind > len(data_to_plot) or ind < 1:
                    error_individual.append(ind)
            if len(error_individual) > 0:
                print("Number of individual is not correct: expected a value between 1 and "+str(len(data_to_plot))+" -> "+str(error_individual))
                return
            individuals = individual
        elif len(self.individual_topsis) > 0:
            individuals = []
            for ind in range(len(self.individual_topsis)):
                for i in range(len(self.individual_topsis[ind])):
                    if i in [0,1]:
                        individuals.append(self.individual_topsis[ind][i])
            individuals = np.unique(np.array(individuals)).tolist()

        ind_values = self.helper_dataframe_to_tuple(value=data_to_plot[['IND']].values.tolist(), individual=individuals)

        
        fig, ax = plt.subplots(figsize=(10,6))

        number_individuals = len(ind_values)
        if current != None:
            number_individuals += 1
            ind_values = ('0',) + ind_values
        
        total = tuple(0 for _ in range(number_individuals))
        chart_data = []
        for c in column_names:
            temp_chart_data = self.helper_dataframe_to_tuple(value=data_to_plot[[c]].values.tolist(), individual=individuals)
            if current != None:
                temp_chart_data = (current[c],) + temp_chart_data    
            chart_data.append(temp_chart_data)
            total = tuple(map(lambda i, j: i + j, total, temp_chart_data))

        for ii in range(len(chart_data)): 
            chart_data[ii] = tuple(map(lambda i, j: (i/j)*100, chart_data[ii], total))
            if ii == 0:
                ax.bar(ind_values, chart_data[ii], width, color=colors[ii], align='center')
                bottom = tuple(0 for _ in range(number_individuals))
            else:
                bottom = tuple(map(lambda i, j: i + j, bottom, chart_data[ii-1]))
                ax.bar(ind_values, chart_data[ii], width, bottom=bottom, color=colors[ii])
                    
        ax.set_xlabel('Individuals')
        ax.set_ylabel('Percentage')
        ax.set_title('Bar Chart technology mix per non domitated solution')
        ax.set_xticks(ind_values)
        ax.legend(labels=column_names, loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.set_ylim(0, 105)

        if show_diversion:
            ax2 = ax.twinx()
            ax2.set_ylim(0, 105)
            diversion = self.helper_dataframe_to_tuple(value=data_to_plot[['Diversion']].values.tolist(), individual=individuals)
            if current_diversion != None:
                diversion = (current_diversion,) + diversion
            ax2.plot(ind_values, diversion, marker='o', color='magenta', linewidth=2)
            ax2.set_yticklabels(tuple('' for _ in range(number_individuals)))
            ax2.legend(labels=['Diversion'], loc='lower right', bbox_to_anchor=(1.25, 1))
        
        plt.show()
        
    def helper_dataframe_to_tuple(self, value, individual=None):
        res = ()
        for ii in range(len(value)):
            if individual != None:
                if ii+1 in individual:
                    res += (value[ii][0],)
            else:
                res += (value[ii][0],)
        return res
    
    def get_history(self):
        """Extract values from res.history of the algorithm

        Returns:
            dict: {n_evals, hist_F, hist_cv, hist_cv_avg}
        """
        if self.has_history == False and self.imported_from_json == False:
            print("No history was saved on the result object of the algorithm")
            return
        return self.history_data
    
    def constraint_satisfaction(self, type):
        """Test the constraint satisfaction of the results and provides plots that illustrate constraint violation

        Args:
            type (string): ['cv_avg', 'cv']
        """
        if self.has_history == False and self.imported_from_json == False:
            print("No history was saved on the result object of the algorithm")
            return
        if self.imported_from_json:
            history = self.history_data
        else:
            history = self.get_history()
        
        n_evals = history['n_evals']

        
        plt.figure(figsize=(7, 5))
        
        if type == 'cv_avg': # Avg. CV of Pop
            # analyze the population
            vals_cv_avg = history['hist_cv_avg']
            k_cv_avg = np.where(np.array(vals_cv_avg) <= 0.0)[0].min()
            plt.plot(n_evals, vals_cv_avg,  color='black', lw=0.7, label="Avg. CV of Pop")
            plt.scatter(n_evals, vals_cv_avg,  facecolor="none", edgecolor='black', marker="p")
            plt.axvline(n_evals[k_cv_avg], color="red", label="All Feasible", linestyle="--")
            plt.suptitle("Convergence")
            plt.title(f"Whole population feasible in Generation {k_cv_avg} after {n_evals[k_cv_avg]} evaluations.", fontsize=10)
            plt.xlabel("Function Evaluations")
            
        elif type == 'cv': # CV of Gen
            # analyze the least feasible optimal solution
            vals_cv = history['hist_cv']
            k_cv = np.where(np.array(vals_cv) <= 0.0)[0].min()
            plt.plot(n_evals, vals_cv,  color='black', lw=0.7, label="CV of Gen")
            plt.scatter(n_evals, vals_cv,  facecolor="none", edgecolor='black', marker="p")
            plt.axvline(n_evals[k_cv], color="red", label="At least one feasible", linestyle="--")
            plt.suptitle("Convergence")
            plt.title(f"At least feasible optimal solution in Generation {k_cv} after {n_evals[k_cv]} evaluations.", fontsize=10)
            plt.xlabel("Function Evaluations")
        else:
            print('Invalid type, types allowed: cv, cv_avg')
            return

        plt.legend()
                
        # Display the plot
        plt.show()

    def line_plot(self, data, x_label, y_label, line_color):
        y_axis = data
        x_axis = list(range(1, len(y_axis)+1))
        
        # Plot
        fig = plt.figure(figsize=(10, 5))
        
        for i in range(len(y_axis)):
            plt.vlines(x=x_axis[i], ymin=0, ymax=y_axis[i], color=line_color) #, label='axvline - % of full height'
        
        plt.xlim(x_axis[0], x_axis[-1]+2)
        plt.ylim(0, np.max(y_axis)+0.2)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def get_topsis(self, weights, ideal_worst=None, ideal_best=None, top_individuals=5, reverse_individuals=False):
        """Implement TOPSIS technique to the results from the pareto front

        Args:
            weights (list): List of weights to be considerd by topsis
            ideal_worst (list, optional): List of ideal worst solution for the objective functions (if known, otherwise topsis calculate these values). Defaults to None.
            ideal_best (list, optional): List of ideal best solution for the objective functions (if known, otherwise topsis calculate these values). Defaults to None.
            top_individuals (int, optional): Number of top individuals to return. Defaults to 5.
            reverse_individuals (bool, optional): Whether or not is desire to plot the top or bottom individuals. Defaults to False.

        Returns:
            dict: Results from topsis {'data': , 'weight': }
            DataFrame: Table with positions (ranks) vs weights showing the individuals and the corresponding topsis score
            list: Ranking of individuals. One ranking per each weight.

        """
        if self.has_result == False:
            print("The optimization doesn't have results, topsis can not be calculated.")
            return
        a = self.res.F        
        if ideal_worst != None:
            if len(a[0]) != len(ideal_worst):
                print("The lenght of ideal_worst must be: "+str(len(a[0]))+", "+str(len(ideal_worst))+" was given.")
        if ideal_best != None:
            if len(a[0]) != len(ideal_best):
                print("The lenght of ideal_best must be: "+str(len(a[0]))+", "+str(len(ideal_best))+" was given.")
        
        sign = [-1, -1, -1]
        result_topsis = []
        result_individuals = []

        column_name = []
        index_name = []        
        for i in range(len(weights)):
            column_name.append('Weight '+str(i+1))
        
        for i in range(len(a)):
            index_name.append('Pos '+str(i+1))
        
        results_d = pd.DataFrame(columns=column_name,
                            index=index_name)

        for w in range(len(weights)):
            res_index, res_a = self.tp.topsis(a, weights[w], sign, custom_ideal_worst=ideal_worst, custom_ideal_best=ideal_best)
            topsy_a = dict()
            for i in range(len(res_a)):
                topsy_a.update({'ind_'+str(i+1):res_a[i]})

            result_topsis.append({'data':res_a, 'weight':', '.join([str(round(elem,2)) for elem in weights[w]])})
            
            temp = sorted(topsy_a.items(), key=lambda x:x[1], reverse=(not reverse_individuals))
            converted_dict = dict(temp)
            
            j = 1
            for cd in converted_dict:
                results_d.at['Pos '+str(j), 'Weight '+str(w+1)] = cd + ': ' + str(round(converted_dict[cd], 6))
                j = j + 1      
            
            individuals = []
            for i in converted_dict.keys():
                individuals.append(int(i.split('_')[1]))   
            result_individuals.append(individuals[0:top_individuals])
        self.result_topsis = result_topsis
        self.individual_topsis = result_individuals
        return result_topsis, results_d, result_individuals
    
    def topsis_plot(self):
        """Plot topsis results obtained by get_topsis method
        """
        if len(self.result_topsis) == 0:
            print("Please, first execute get_topsis function with the weights")
            return
        colors = {0:'gold',1:'darkorange',2:'blue',3:'limegreen',4:'red',5:'violet', 6:'green', 7:'brown'}
        x_axis = list(range(1,len(self.result_topsis[0]['data'])+1))

        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_xlabel('Individuals')
        ax.set_ylabel('Closeness degree (%)')
        #ax.set_title('')
        ax.set_xticks(x_axis)
        ax.set_ylim(0, 105)

        data_percentage = list(map(lambda dd: dd['data']*100, self.result_topsis))
        for d in range(len(data_percentage)):
            ax2 = ax.twinx()
            ax2.set_ylim(0, 105)
            ax2.plot(x_axis, data_percentage[d], marker='o', color=colors[d], linewidth=2, label='Weight '+str(d+1)+': '+self.result_topsis[d]['weight'])
            ax2.set_yticklabels([])

        fig.legend(loc='upper right', bbox_to_anchor=(1.2, 1)) 

        plt.show()  

    def running_metric(self, delta_gen, n_plots, gen=None):
        """Plots for running metric
        Args:
            delta_gen (int): distance between generations to be plotted 
            n_plots (int): number of plots 
            gen (int, optional): max generation to be plotted. Defaults to None.
        """
        if self.has_history == False:
            print("No history was saved on the result object of the algorithm")
            return

        if n_plots * delta_gen > len(self.running_data):
            n_plots = round(len(self.running_data)/delta_gen)

        figure, ax1 = plt.subplots(n_plots, figsize=(6,n_plots*4))
        data = []
        generation = delta_gen-1
        if gen != None:
            if gen > len(self.res.algorithm.callback.running_history["history"]) or gen < 1:
                print("Generation "+str(gen)+" is out of range, value allowed between 1 and "+str(len(self.running_history["history"])))
                return
            else:
                generation = gen-1
                n_plots = 1
         
        col_size = int(math.ceil(n_plots / 15))
        for i in range(0, n_plots):
            if generation < len(self.running_data):
                if n_plots == 1:
                    ax = ax1
                else:
                    ax = ax1[i]
                data.append(self.running_data[generation])

                for tau, x, f, v in data[:-1]:
                    ax.plot(x, f, label="t=%s" % tau, alpha=0.6, linewidth=3)

                tau, x, f, v = data[-1]
                ax.plot(x, f, label="t=%s (*)" % tau, alpha=0.9, linewidth=3)
        
                for k in range(len(v)):
                    if v[k]:
                        ax.plot([k + 1, k + 1], [0, f[k]], color="black", linewidth=0.5, alpha=0.5)
                        ax.plot([k + 1], [f[k]], "o", color="black", alpha=0.5, markersize=2)
                ax.set_yscale("symlog")
                ax.legend(bbox_to_anchor=(1.07+(0.2*col_size), 1), ncol=col_size)
        
                ax.set_xlabel("Generation")
                ax.set_ylabel("$\Delta \, f$", rotation=0)
                generation += delta_gen
        self.igd = []
        for rd in data:
            self.igd.append({'tau':rd[0], 'igd':rd[2][-2]})
    
    def plot_sankey(self, fileName, individual):
        if self.has_result == False:
            print("NO RESULTSET TO CREATE A PLOT")
            return 
        result = self.res.X
        total_individual = len(result)
        if individual > total_individual or individual < 1:
            print("Number of individual is not correct: expected a value between 1 and "+str(total_individual)+" -> "+str(individual))
            return
        optElem = self.optimization_list[0]
        optElem.optimized_x = list()
        for i in range(len(optElem.project.parameters_list)):
            optElem.optimized_x.append({'name': optElem.project.parameters_list[i]['name'],
                        'amount':result[individual-1][i]})   
        for k, v in optElem.scheme_vars_dict.items():
            optElem.optimized_x.append({'name': v,
                        'amount': result[individual-1][k]})
        optElem.plot_sankey(fileName=fileName)

    def export_to_json(self, filename):
        if self.imported_from_json and self.has_result:
            print("No data to export")
            return
        try:
            for i in range(len(self.running_data)):
                for j in range(len(self.running_data[i][3])):
                    if self.running_data[i][3][j]:
                        self.running_data[i][3][j] = 1
                    else:
                        self.running_data[i][3][j] = 0

            export_data = {'method_list':self.method_list, 'X':self.res.X, 'F':self.res.F, 'G':self.res.G, 'history_data':self.get_history(), 'running_data':self.running_data, 'running_history':self.running_history, 'igd':self.igd}
            
            json_str = json.dumps(export_data, indent=2, cls=NumpyArrayEncoder)
            with open(filename + ".json", "w") as write_file:
                write_file.write(json_str)
            print("Data was exported successfully to the json file called: " + filename + ".json!!!")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("Data was not exported to json.!!!")

    def import_from_json(self, filename):
        try:
            # JSON file
            f = open (filename + '.json', "r")
            
            # Reading from file
            data = json.loads(f.read())

            self.res = Result()
            self.res.X = data['X']
            self.res.F = data['F']
            self.res.G = data['G']
            self.history_data = data['history_data']
            self.running_data = data['running_data']
            self.running_history = data['running_history']
            self.igd = data['igd']   

            for i in range(len(data['method_list'])):
                data['method_list'][i]['method'] = tuple(row for row in data['method_list'][i]['method'])

            self.method_list = data['method_list']
            
            self.has_result = True
            self.imported_from_json = True
            
            # Closing file
            f.close()
            print("Data was imported successfully from the json file called: " + filename + ".json!!!")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("Data was not imported from json.!!!")
    
    def import_from_csv(self, filename):
        try:
            df = pd.read_csv(filename + '.csv', index_col=[0])
            temp = df.T
            temp['IND'] = temp['IND'].fillna(0)
            temp = temp.astype({'IND': 'string'})
            temp['IND'] = temp['IND'].apply(lambda r:r.split('.')[0])
            self.results = temp.T
            print("Data was imported successfully from the csv file called: " + filename + ".csv!!!")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("Data was not imported from csv.!!!")

class HistoryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.running_history = {'history':[], 'delta_nadir':[], 'delta_ideal':[]}
        self.running_data = []
        self.history_data = { 'n_evals': [], 'hist_F': [], 'hist_cv': [], 'hist_cv_avg': []}

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        c_F, c_ideal, c_nadir = F, F.min(axis=0), F.max(axis=0)

        # the current norm that should be used for normalization
        norm = c_nadir - c_ideal
        norm[norm < 1e-32] = 1.0

        # normalize the current objective space values
        c_N = normalize(c_F, c_ideal, c_nadir)

        # normalize all previous generations with respect to current ideal and nadir
        N = [normalize(e["F"], c_ideal, c_nadir) for e in self.running_history['history']]
        
        # append the current optimum to the history
        self.running_history['history'].append(dict(F=F, ideal=c_ideal, nadir=c_nadir))

        self.running_history['delta_ideal'] = [calc_delta_norm(self.running_history['history'][k]["ideal"], self.running_history['history'][k-1]["ideal"], norm) for k in range(1, len(self.running_history['history']))] + [0.0]
        self.running_history['delta_nadir'] = [calc_delta_norm(self.running_history['history'][k]["nadir"], self.running_history['history'][k-1]["nadir"], norm) for k in range(1, len(self.running_history['history']))] + [0.0]

        delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]

        tau = algorithm.n_gen
        f = delta_f
        x = np.arange(len(f)) + 1
        v = [max(ideal, nadir) > 0.005 for ideal, nadir in zip(self.running_history['delta_ideal'], self.running_history['delta_nadir'])]
       
        self.running_data.append((tau, x, f, v))

        # store the number of function evaluations
        self.history_data["n_evals"].append(algorithm.evaluator.n_eval)
    
        # retrieve the optimum from the algorithm
        optmulti= algorithm.opt
    
        # store the least contraint violation and the average in each population
        self.history_data["hist_cv"].append(optmulti.get("CV").min())
        self.history_data["hist_cv_avg"].append(algorithm.pop.get("CV").mean())
    
        # filter out only the feasible and append and objective space values
        feas = np.where(optmulti.get("feasible"))[0]
        self.history_data["hist_F"].append(optmulti.get("F")[feas])

class DummyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def notify(self, algorithm):
        return

class FractionSumOneRepair(Repair):
    """Guarantees a faster feasibility of the equality constraints

    Args:
        Repair (pymoo.core.repair): Repair class of Pymoo Library
    """
    def __init__(self, equality_var_array, verbose = False):
        self.equality_var_array = equality_var_array
        self.verbose = verbose
        super().__init__()

    def _do(self, problem, Z, **kwargs):

        # sum of parameters in each group should be one
        Q = 1

        if self.verbose:
            print("******************INITIAL Z:")
            print(Z)
        # now repair each indvidiual i
        for i in range(len(Z)):
            z = Z[i]
            index = 0
            for j in self.equality_var_array: 
                N_param_Ingroup = self.equality_var_array[j]
                sum_param_Ingroup = sum(z[k] for k in range(index, index + N_param_Ingroup))
                if (np.isclose(sum_param_Ingroup, Q, rtol=1e-04) == False):
                    for k in range(index, index + N_param_Ingroup):
                        z[k] = z[k] / sum_param_Ingroup
                index += N_param_Ingroup 
        if self.verbose:
            print("******************FINAL Z:")
            print(Z)
        return Z

class NumpyArrayEncoder(JSONEncoder):
    """Helper class to transform list into json compatible list

    Args:
        JSONEncoder (json): JSONEncoder object
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)