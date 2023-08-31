# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:22:00 2023

@author: jepiguti
"""
from .Optimization import Optimization
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from copy import deepcopy
import json
from multiprocessing import Pool, cpu_count, TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import os
import pyDOE
from time import time
from brightway2 import projects
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

class MultiOptimization():
    """

    :param functional_unit:
    :type functional_unit: dict
    :param method:
    :type method: list
    :param project:
    :type project: ``swolfpy.Project.Project``

    """
    def __init__(self, functional_unit, method, project):
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
                                 termination=40, seed=1, verbose=True, 
                                 repair=False, verbose_repair=False):
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
        """
        Minimize
        """  
        res = MultiObjMinimize(problem=problem,
                                algorithm=algorithm,
                                termination=('n_gen', termination),
                                seed=seed,
                                verbose=verbose)

        return res
    
    def getScatter(values):        
        Scatter().add(values).show()

    

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