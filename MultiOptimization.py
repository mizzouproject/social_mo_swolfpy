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
#pymoo imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as MultiObjMinimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.visualization.scatter import Scatter
from pymoo.core.repair import Repair

import shutup; shutup.please()
from datetime import datetime

import matplotlib.pyplot as plt
import oapackage
from oapackage.oahelper import create_pareto_element
from sklearn import preprocessing

from pymoo.util.running_metric import RunningMetric
from pymoo.util.running_metric import RunningMetricAnimation

class MultiOptimization():
    """

    :param functional_unit:
    :type functional_unit: dict
    :param method:
    :type method: list
    :param project:
    :type project: ``swolfpy.Project.Project``
    :type bw2_fu: bw2data.backends.peewee.proxies.Activity

    """
    def __init__(self, functional_unit, method, project, bw2_fu):
        self.optimization_list = []
        self.configArray = []
        self.objs = []
        self.constr_eq = []
        self.constr_ieq = []
        self.cons = []
        self.n_var = 0
        for m in method:
            mth = []
            mth.append(m)
            tmpOpt = Optimization(functional_unit=functional_unit,method=mth,project=project)
            self.objs.append(tmpOpt._objective_function)
            self.optimization_list.append(tmpOpt)
        #self.config(project)
        self.res = [] #to save the result of the algorithm
        self.bw2_fu = bw2_fu
        self.has_result = False #boolean to know if the algorithm has a result (res.X exists)
        self.results = [] # to save the DataFrame of the function report_res()
        self.has_history = False
        self.running_data = None
        self.igd = []
    
    def config(self, project):
        for optElem in self.optimization_list:
            self.configArray = optElem.config(project)
    
    def set_config(self):
        for optElem in self.optimization_list:
            optElem.set_config(self.configArray)            
        self.n_var = len(self.optimization_list[0].project.parameters_list) + self.optimization_list[0].n_scheme_vars

    def set_constraints(self, constraints=None, collection=False):
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
        number_of_results = 0
        self.results = []
        self.has_history = save_history
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
        n_var       ->
        objs        ->
        constr_ieq  ->
        constr_eq   ->
        xl          ->
        xu          ->
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
                    crossover=SBX(),
                    mutation=PolynomialMutation(),
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
                                save_history=save_history)
        end_time = datetime.now() # current date and time
        end_date_time = end_time.strftime("%H:%M:%S")
        print("Optimization finished at: "+end_date_time)
        total_time = datetime.strptime(end_date_time, "%H:%M:%S") - datetime.strptime(start_date_time, "%H:%M:%S")
        print(f"Optimization lasted: {total_time.total_seconds()} seconds")
        try:
            number_of_results = len(res.X)
            self.has_result = True
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        finally:
            print('Resulset Size = '+str(number_of_results))
            self.res = res
        return res
    
    def report_res(self, Opt=False, fileName='', update=False):
        if self.has_result == False:
            print("NO RESULTSET TO PROCESS A REPORT")
        else:
            if Opt == False:
                Opt = self.optimization_list[0]
            
            if len(self.results) == 0 or update:
                results = pd.DataFrame()
                for individual in range(len(self.res.X)):
                    Opt._objective_function(self.res.X[individual])            
                    individual_name = "ind_"+str(individual+1)
                    if 'unit' in results.columns:
                        results.insert(len(results.columns),individual_name,np.nan)
                    else:
                        results = pd.DataFrame(columns=['unit'],
                                        index=['Diversion','GWP','Operation Cost','Capital cost','Total cost','Social cost',
                                                'Collection','LF','WTE','AD','Composting','SS_MRF','Reprocessing','IND'])
                        results.at['GWP', 'unit'] = 'kg CO2/Mg'
                        results.at['Operation Cost', 'unit'] = '$/Mg'
                        results.at['Capital cost', 'unit'] = '$/Mg'
                        results.at['Total cost', 'unit'] = '$/Mg'
                        results.at['Social cost', 'unit'] = '$/Mg'
                        results.at['Diversion', 'unit'] = '%'
                        for process in ['Collection','LF','WTE','AD','Composting','SS_MRF','Reprocessing']:
                            results.at[process, 'unit'] = 'Mg/yr'
                    
                    results.at['IND', individual_name] = str(individual+1)
                    
                    Opt.switch_method(('CML 2001 (obsolete)', 'climate change', 'GWP 100a'))
                    Opt.lcia()
                    results.at['GWP', individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)

                    Opt.switch_method(('SwolfPy_Operational_Cost', 'SwolfPy'))
                    Opt.lcia()
                    results.at['Operation Cost', individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)

                    Opt.switch_method(('SwolfPy_Capital_Cost', 'SwolfPy'))
                    Opt.lcia()
                    results.at['Capital cost', individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)

                    Opt.switch_method(('SwolfPy_Total_Cost', 'SwolfPy'))
                    Opt.lcia()
                    results.at['Total cost', individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)
                    
                    Opt.switch_method(('SwolfPy_Social','SwolfPy'))
                    Opt.lcia()
                    results.at['Social cost', individual_name] = round(Opt.score/float(self.bw2_fu.as_dict()['unit'].split(' ')[0]),2)

                    for process in ['Collection','LF','WTE','AD','Composting','SS_MRF','Reprocessing']:
                        results.at[process, individual_name] = round(LCA_matrix.get_mass_flow(Opt, process))

                    results.at['Diversion', individual_name] = round((1-results[individual_name]['LF']/results[individual_name]['Collection'])*100,2)
                self.results = results
            else:
                results = self.results

            if len(fileName) > 0:
                results.to_csv(fileName+'.csv')
        return(results)

    def test_pareto(self, value=None, show=False):
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
            values = [[datapoints[0,ii]], [datapoints[1,ii]], [datapoints[2,ii]]]
            val = create_pareto_element(values, pareto=pareto)
            pareto.addvalue(val, ii)

        if show:
            pareto.show(verbose=1)
        return pareto

    def dynamic_scatter_plot_2D(self, x_name, y1_name, y2_name):
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
        
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(y1_name, color='g')
        ax2.set_ylabel(y2_name, color='b')
        
        # Display the plot
        plt.show() 

    def dynamic_scatter_plot_3D(self, x_name, y_name, z_name):
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
        d3 = data_to_plot[[x_name, y_name, z_name]]
        d4 = preprocessing.normalize(d3)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')

        # Data for three-dimensional scattered points
        zdata = d4.T[2]
        xdata = d4.T[0]
        ydata = d4.T[1]
        
        ax.scatter3D(xdata, ydata, zdata); #, c=zdata
    
        #ax.set_box_aspect(aspect = (1,1,1))
        #ax.text(.5, .5, .5, s='some string')
        
        ax.set_xlabel(x_name, fontsize=10, labelpad=13)
        ax.set_ylabel(y_name, fontsize=10, labelpad=13)
        ax.set_zlabel(z_name, fontsize=10, labelpad=3, rotation= 'vertical')

        plt.show()
    
    def dynamic_multi_scatter_plot_2D(self, column_names):
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
            ax.set(xlabel=c[0], ylabel=c[1])
            i += 1
            
        figure.tight_layout()
        
        # Display the plot
        plt.show()

    def dynamic_bar_chart_tech_mix(self, current=None, column_names=None, individual=None, current_diversion=None, show_diversion=False):
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
        if individual != None:
            for ind in individual:
                if ind > len(data_to_plot) or ind < 1:
                    error_individual.append(ind)
            if len(error_individual) > 0:
                print("Number of individual is not correct: expected a value between 1 and "+str(len(data_to_plot))+" -> "+str(error_individual))
                return
        ind_values = self.helper_dataframe_to_tuple(value=data_to_plot[['IND']].values.tolist(), individual=individual)

        fig, ax = plt.subplots(figsize=(10,6))

        number_individuals = len(ind_values)
        if current != None:
            number_individuals += 1
            ind_values = ('0',) + ind_values
        
        total = tuple(0 for _ in range(number_individuals))
        chart_data = []
        for c in column_names:
            temp_chart_data = self.helper_dataframe_to_tuple(value=data_to_plot[[c]].values.tolist(), individual=individual)
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
            diversion = self.helper_dataframe_to_tuple(value=data_to_plot[['Diversion']].values.tolist(), individual=individual)
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
        if self.has_history == False:
            print("No history was saved on the result object of the algorithm")
            return
            
        history = self.res.history  # extract information from save history flag
        
        n_evals = []             # corresponding number of function evaluations\
        hist_F = []              # the objective space values in each generation
        hist_cv = []             # constraint violation in each generation
        hist_cv_avg = []         # average constraint violation in the whole population
        
        for algo in history:
        
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)
        
            # retrieve the optimum from the algorithm
            optmulti= algo.opt
        
            # store the least contraint violation and the average in each population
            hist_cv.append(optmulti.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())
        
            # filter out only the feasible and append and objective space values
            feas = np.where(optmulti.get("feasible"))[0]
            hist_F.append(optmulti.get("F")[feas])

        return {
                    'n_evals': n_evals,             
                    'hist_F': hist_F,              
                    'hist_cv': hist_cv,
                    'hist_cv_avg': hist_cv_avg
                }
    
    def constraint_satisfaction(self, type):
        if self.has_history == False:
            print("No history was saved on the result object of the algorithm")
            return
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
    
    def running_metric(self, delta_gen, n_plots, gen=None):
        if self.has_history == False:
            print("No history was saved on the result object of the algorithm")
            return
        running = RunningMetricAnimation(delta_gen=delta_gen,
                        n_plots=n_plots,
                        key_press=False,
                        do_show=True)
        if gen != None:
            if gen > len(self.res.history) or gen < 1:
                print("Generation "+str(gen)+" is out of range, value allowed between 1 and "+str(len(self.res.history)))
                return
            else:
                history = self.res.history[:gen]
        else:
            history = self.res.history

        for algorithm in history:
            running.update(algorithm)

        self.running_data = running.data

        self.igd = []
        for rd in running.data:
            self.igd.append({'tau':rd[0], 'igd':rd[2][-2]})
   
class FractionSumOneRepair(Repair):

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