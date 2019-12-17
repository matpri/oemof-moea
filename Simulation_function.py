# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:28:42 2018

@author: VCasalicchio, MPrina
"""
"""
General description
-------------------
Energy system model using oemof

Data
----
scenario2015.xlsx
"""
from gurobipy import *
import os
import pandas as pd
import numpy as np
from oemof import solph
from oemof import outputlib
from termcolor import colored
from datetime import datetime
from liboemof import nodes_from_excel, create_nodes, costs, co2
from random import randint
from deap import base, creator
import random
from deap import tools
import multiprocessing
import csv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
print('1')

global xls, MOLT_FACTORS, VARIABLES, count, count2, DatiFuel

start_time = datetime.now()
###-----------------------INPUT DATA-------------------------------------------
data = {"filename_RS":"scenario2015.xlsx",
        "number of processes": 8,
        "Genetic algorithm": {"Size of population": 36,
                              "Number of generations": 2},
        "Variables": [{"label": "R1_solar", "target": 0, "Range": [8, 55],"Moltiplication factor": 1000},
                      {"label": "R1_wind", "target": 0,"Range": [1, 6],"Moltiplication factor": 100},
                      {"label": "R2_solar", "target": 0,"Range": [2, 12],"Moltiplication factor": 1000},
                      {"label": "R2_wind", "target": 0,"Range": [1, 7],"Moltiplication factor": 100},
                      {"label": "R3_solar", "target": 0,"Range": [2, 26],"Moltiplication factor": 1000},
                      {"label": "R3_wind", "target": 0,"Range": [1, 8],"Moltiplication factor": 1000},
                      {"label": "R4_solar", "target": 0,"Range": [3, 14],"Moltiplication factor": 1000},
                      {"label": "R4_wind", "target": 0,"Range": [4, 23],"Moltiplication factor": 1000},
                      {"label": "R5_solar","target": 0, "Range": [7, 33],"Moltiplication factor": 100},
                      {"label": "R5_wind", "target": 0,"Range": [1, 5],"Moltiplication factor": 1000},
                      {"label": "R6_solar", "target": 0,"Range": [1, 10],"Moltiplication factor": 1000}, 
                      {"label": "R6_wind", "target": 0,"Range": [2, 10],"Moltiplication factor": 1000}, 
                      {"label": "R1_storage_batt", "target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R2_storage_batt", "target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R3_storage_batt", "target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R4_storage_batt","target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R5_storage_batt","target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R6_storage_batt","target": 0,"Range": [0, 10],"Moltiplication factor": 10000},
                      {"label": "R1_R2_powerline", "target": "capacity_1", "Range": [33, 100],"Moltiplication factor": 100},
                      {"label": "R1_R2_powerline", "target": "capacity_2","Range": [23, 100],"Moltiplication factor": 100},
                      {"label": "R2_R3_powerline", "target": "capacity_1","Range": [20, 100],"Moltiplication factor": 100},
                      {"label": "R2_R3_powerline", "target": "capacity_2","Range": [25, 100],"Moltiplication factor": 100},
                      {"label": "R3_R4_powerline", "target": "capacity_2","Range": [38, 100],"Moltiplication factor": 100},
                      {"label": "R3_R5_powerline", "target": "capacity_1","Range": [9, 100],"Moltiplication factor": 100},
                      {"label": "R3_R5_powerline", "target": "capacity_2","Range": [7, 100],"Moltiplication factor": 100},
                      {"label": "R4_R6_powerline", "target": "capacity_1","Range": [11, 100],"Moltiplication factor": 100},
                      {"label": "R4_R6_powerline", "target": "capacity_2","Range": [10, 100],"Moltiplication factor": 100}]
        }
        
VARIABLES = tuple([(dic['label'], dic['target']) for dic in data["Variables"]])
#print(VARIABLES)
X = [tuple(dic['Range']) for dic in data["Variables"]]
print (X)
MOLT_FACTORS = [dic['Moltiplication factor'] for dic in data["Variables"]]
#print (MOLT_FACTORS)

filename_RS =data['filename_RS']
#filename_seq =data['filename_seq']
#print(filename)
Npop= data['Genetic algorithm']["Size of population"]
#print (NPOP)
Ngen= data["Genetic algorithm"]["Number of generations"]

xls = pd.ExcelFile(filename_RS)

DatiFuel=pd.ExcelFile('DatiFuel.xlsx')

DatiCosti = pd.ExcelFile('Dati_Costi.xlsx').parse(sheet_name="database_costs")

def flush():
    pass
###-----------------------SIMULATION-------------------------------------------
def Simulation(individual):
    print(individual)
    var = [i*j for i,j in list(zip(MOLT_FACTORS, individual))]
    
    name_ind='_'.join(map(str, individual))
    resultfile='results/energy_system_%s.csv'%name_ind
    dic_output={}
    
    if os.path.exists(resultfile):
        dic_output={}
        with open(resultfile, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dic_output=row
                
    else: 
        
        excel_nodes = nodes_from_excel(xls, var, VARIABLES)

        datetime_index = pd.date_range('2015-01-01 00:00:00','2015-01-07 23:00:00',freq='60min')
        
        esys = solph.EnergySystem(timeindex=datetime_index)
        my_nodes = create_nodes(nd=excel_nodes)
        esys.add(*my_nodes)
#        # creation of a least cost model from the energy system
        om = solph.Model(esys)
        om.receive_duals()  
#        # solving the linear problem using the given solver
        om.solve(solver='gurobi')#, solve_kwargs={'tee': False}
#        
        results = outputlib.processing.results(om)
        #print(results)
    
        #outputlib.processing.param_results(esys, exclude_none=True) 
        outputlib.processing.parameter_as_dict(esys, exclude_none=True)
            
        f1 = outputlib.views.node(results, 'R1_bus_el')['sequences'].sum()
        f2 = outputlib.views.node(results, 'R2_bus_el')['sequences'].sum()
        f3 = outputlib.views.node(results, 'R3_bus_el')['sequences'].sum()
        f4 = outputlib.views.node(results, 'R4_bus_el')['sequences'].sum()
        f5 = outputlib.views.node(results, 'R5_bus_el')['sequences'].sum()
        f6 = outputlib.views.node(results, 'R6_bus_el')['sequences'].sum()
        
        
        f7 = outputlib.views.node(results, 'R1_bus_th')['sequences'].sum()
        f8 = outputlib.views.node(results, 'R2_bus_th')['sequences'].sum()
        f9 = outputlib.views.node(results, 'R3_bus_th')['sequences'].sum()
    
    #    print('end linear opt')
        
        Costs=costs(DatiCosti, excel_nodes,f1,f2,f3,f4,f5,f6)
        
#------------------------------------------------------------------------------
#       +PowerlinesCosts
#------------------------------------------------------------------------------        
        Cost_per_MW_200_km_y = 0.237
        varr= var[-9:]
        costs_powerline = sum([varr[j]*Cost_per_MW_200_km_y for j in range(len(varr))])
        Costs=costs_powerline+Costs
#------------------------------------------------------------------------------     

        CO2=co2(DatiFuel, excel_nodes, f1,f2,f3,f4,f5,f6,f7,f8,f9)

        dic_output={}
        dic_output['CO2']=CO2
        dic_output['COSTS']=Costs         
#        collection[name_ind] = dic_output
        with open(resultfile, 'w') as fl:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(fl, dic_output.keys())
            w.writeheader()
            w.writerow(dic_output)

    CO2 =float(dic_output['CO2'])#collection[name_ind]['CO2']
    Costs =float(dic_output['COSTS'])#collection[name_ind]['COSTS']
    print('CO2 [Mt]', colored(CO2, 'red'), 'costs [M€]', colored(Costs, 'red'))
    return CO2, Costs


'''Definition of objectives, should be a tuple -1 for minimization and +1 for maximization
you can set also weights, see fitness manual of DEAP'''
objectives = (-1.0, -1.0)

min_b = list(zip(*X))[0]
max_b = list(zip(*X))[1]
#print(min_b, max_b)

creator.create("FitnessMin", base.Fitness, weights=objectives)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

list_attr = []
for i, bnd in enumerate(X):
    attr = 'attr_l%i' % i
    toolbox.register(attr, random.randint, bnd[0], bnd[1])
    list_attr.append(toolbox.__getattribute__(attr))

toolbox.__dict__.keys()
#new_multi
#    toolbox.register("map", futures.map)
#    pool = multiprocessing.Pool(processes=2)
#    toolbox.register("map", pool.map)

#creator.create("individual", ind_guess1, fitness=creator.FitnessMin)
#toolbox.register("individual", ind_guess2, creator.Individual)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 list_attr, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxUniform,
                  indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt,
                 low=min_b, up=max_b, indpb=5.0/100)                    
#    toolbox.register("mutate", tools.mutPolynomialBounded,
#                     low=min_b, up=max_b, eta=1.0, indpb=1.0/100)
toolbox.register("select", tools.selNSGA2)
print('4')

def GA(toolbox, evaluate, n_pop, n_gen, feasible=None, penalty=None):
    """ Excute the GA algorithms.

    :x:  range of variables to create a random grid
    :evaluate: function to evaluate
    :weights: negative minimization, positive maximization
    :n: size of population
    :ngen: number of generations
    :[feasible]: function for boundary constraints
    """
    toolbox.register("evaluate", evaluate)

    if feasible:
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, penalty))

    pop = toolbox.population(n=n_pop)
    
    from Seed import seed_list
    for ind in seed_list:
        del pop[0]
        
        guess_ind = creator.Individual(ind)    
        pop.append(guess_ind)    
    
    pop0 = pop
#    print (pop, len(pop))
    pop_back_up = pop[:]

    # Evaluate the entire population
#    fitnesses = list(map(toolbox.evaluate, pop))
    '''different from no_multi version'''
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    ff0 = fitnesses
#    print (colored(ff0, 'red'))
    
    for ind, fit in list(zip(pop, fitnesses)):
        ind.fitness.values = fit
        
    for ind, fit in list(zip(pop_back_up, fitnesses)):
        ind.fitness.values = fit 
#    pop_hist=[]
#    fit_hist=[]
#    pop_hist.append(pop0)
#    fit_hist.append(ff0)
    hist = {'population': {}, 'fitness': {}}
    hist['population'][0] = list(pop0) #list(zip(*pop0))
    hist['fitness'][0] = list(ff0) #list(zip(*ff0))

    pop = toolbox.select(pop, len(pop))

    for gen in range(1, n_gen):
        print('step: ', colored(gen, 'red'))
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in list(zip(offspring[::2], offspring[1::2])):
            if random.random() <= 0.9:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        invalids = [indiv for indiv in offspring if not indiv.fitness.valid]
#        print('invalids', invalids)
        
        for ind in offspring:
            if ind in invalids:    
                if ind in pop_back_up:
#                    print('YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS', ind)
                    for ind2 in pop_back_up:
                        if ind2==ind:
#                            print('found', ind2.fitness.values )
                            ind.fitness.values = ind2.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in list(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit

        for ind in invalid_ind:
            pop_back_up.append(ind)

        pop = toolbox.select(pop + offspring, n_pop)
#        print (pop)
#        pop_hist.append(pop)
        fitnesses = [ind.fitness.values for ind in pop]
        hist['population'][gen] = list(pop) #list(zip(*pop))
#        fitnesses = [ind.fitness.values for ind in pop]
        hist['fitness'][gen] = list(fitnesses)#list(zip(*fitnesses))
#        fit_hist.append(fitnesses)
#        print(fitnesses)
        #print (colored(fitnesses, 'blue'))
    ff=fitnesses
    
    return (pop0, pop), (ff0, ff), hist

'''4) pop is a tuple with the initial and final population
ff contains the fitness values of the initial and final population
hist is a dictionary: {'populations': pop_hist, 'fitness': fit_hist} containing all the individuals evaluated and all the fitnesses'''
#pool = multiprocessing.Pool(processes=2)
if __name__ == "__main__":
#    toolbox = base.Toolbox()
    print('5')
    pool = multiprocessing.Pool(processes=data['number of processes'])
    toolbox.register("map", pool.map)
#    val = data["Genetic algorithm"][0]
    pop = toolbox.population(n=Npop)
    pop, ff, hist = GA(toolbox, Simulation, Npop, Ngen)
    
    print(ff)

    index=[]
    for dic in data["Variables"]:
       if dic['target']==0:
            index.append(dic['label'])
       else:
           index.append(dic['label']+str('_')+dic['target'])
    index.append('CO2 [Mt]')        
    index.append('Costs [M€]')        
    
    mf=[]
    for dic in data["Variables"]:
            mf.append(dic['Moltiplication factor'])
            
    frame=[]
    for j in range (Ngen):
        for i in range(len(hist['population'][j])):
            ind=[]
            ind=[a*b for a,b in zip(list(hist['population'][j][i]),mf)]
            #ind.append(ind)
        #    [a*b for a,b in zip(a,b)]
            ind.append(hist['fitness'][j][i][0])
            ind.append(hist['fitness'][j][i][1])
            frame.append(ind)    
        
    df=pd.DataFrame(data=frame, index=None, columns=index, dtype=None, copy=False)
    writer = pd.ExcelWriter('SCENARIOS.xlsx')
    df.to_excel(writer,'scenarios')
    writer.save()         


end_time = datetime.now()
print('\nDuration: {}'.format(end_time - start_time))











