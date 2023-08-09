# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:22:00 2023

@author: jepiguti
"""
from .Optimization import Optimization
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
            
    def multi_obj_optimization(self, constraints=None, collection=False, n_iter=30,
                                 nproc=None, timeout=None, initialize_guess='random'):
        """
        Initialization of each Optimization object inside the optimization_list
        """
        for optElem in self.optimization_list:
            optElem.oldx = [0 for i in range(self.n_var)]
            optElem.magnitude = len(str(int(abs(optElem.score))))
            optElem.constraints = constraints
            optElem.collection = collection
            optElem.cons = optElem._create_constraints()
        self.cons = self.optimization_list[0].cons
    
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
        """
        Setup of the Algorithm
        """  

        """
        Minimize
        """  

    
    def get_variables(self):
        print(self.n_var)
        print("***************************")
        print(self.constr_eq)
        print("***************************")
        print(self.constr_ieq)