# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:54:40 2018

@author: MPrina
"""
import os
import logging
import pandas as pd
import numpy as np

import math

from oemof.tools import logger
from oemof import solph
from oemof import outputlib
#from oemof.graph import create_nx_graph
#from matplotlib import pyplot as plt
#import networkx as nx
from termcolor import colored

from Electric_cost_function import selectric

from datetime import datetime
start_time = datetime.now()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# nodes - Oemof
def nodes_from_excel(xls, ind, VARIABLES):
    """Read node data from Excel sheet

    Parameters
    ----------
    filename : :obj:`str`
        Path to excel file

    Returns
    -------
    :obj:`dict`
        Imported nodes data
    """

    nodes_data = {'buses': xls.parse('buses'),
                  'commodity_sources': xls.parse('commodity_sources'),
                  'transformers_pp': xls.parse('transformers_pp'),
                  'transformers_chp': xls.parse('transformers_chp'),
                  'renewables': xls.parse('renewables'),
                  'demand': xls.parse('demand'),                 
                  'storages': xls.parse('storages'),
                  'powerlines': xls.parse('powerlines'),
                  'timeseries': xls.parse('time_series')
                  }

    # set datetime index
#    nodes_data['timeseries'].set_index('timestamp', inplace=True)
#    nodes_data['timeseries'].index = pd.to_datetime(
#        nodes_data['timeseries'].index)
    
    for a in range(len(VARIABLES)):
        if a < 12:
#            print(a)
            nodes_data['renewables'].loc[nodes_data['renewables'].loc[nodes_data['renewables']['label']==VARIABLES[a][0]].index, 'capacity']=ind[a]
#            print(nodes_data['renewables'].loc[nodes_data['renewables'].loc[nodes_data['renewables']['label']==VARIABLES[a][0]].index, 'capacity'], ind[a])
        if a>=12 and a<18:
            nodes_data['storages'].loc[nodes_data['storages'].loc[nodes_data['storages']['label']==VARIABLES[a][0]].index, 'nominal capacity']=ind[a]
            nodes_data['storages'].loc[nodes_data['storages'].loc[nodes_data['storages']['label']==VARIABLES[a][0]].index, 'capacity inflow']=ind[a]
            nodes_data['storages'].loc[nodes_data['storages'].loc[nodes_data['storages']['label']==VARIABLES[a][0]].index, 'capacity outflow']=ind[a]
            
        if a>=18 and a<27:
            if 'capacity_1' in VARIABLES[a][1]:
                nodes_data['powerlines'].loc[nodes_data['powerlines'].loc[nodes_data['powerlines']['label']==VARIABLES[a][0]].index, 'capacity_1']=ind[a]
            elif 'capacity_2' in VARIABLES[a][1]:
                nodes_data['powerlines'].loc[nodes_data['powerlines'].loc[nodes_data['powerlines']['label']==VARIABLES[a][0]].index, 'capacity_2']=ind[a]

    """
    nodes_data['hp_av'].index = pd.to_datetime(
        nodes_data['timeseries'].index)
    nodes_data['tr_av'].index = pd.to_datetime(
        nodes_data['timeseries'].index)
    """    
    return nodes_data


def create_nodes(nd=None):

    if not nd:
        raise ValueError('No nodes data provided.')

    nodes = []

    # Create Bus objects from buses table
    busd = {}

    for i, b in nd['buses'].iterrows():

        if b['active']:
            bus = solph.Bus(label=b['label'])
            nodes.append(bus)

            busd[b['label']] = bus
            
            
            if b['excess']:
                nodes.append(
                    solph.Sink(label=b['label'] + '_excess',
                               inputs={busd[b['label']]: solph.Flow(
                                   variable_costs=b['excess costs'])})
                )
            if b['shortage']:
                nodes.append(
                    solph.Source(label=b['label'] + '_shortage',
                                 outputs={busd[b['label']]: solph.Flow(
                                     variable_costs=b['shortage costs'])})
                     )            


    # Create Source objects from table 'commodity sources'
    for i, cs in nd['commodity_sources'].iterrows():
        if cs['active']:
            nodes.append(
                solph.Source(label=cs['label'],
                             outputs={busd[cs['to']]: solph.Flow(
                                 variable_costs=cs['variable costs'])})
                        )

    # Create Source objects with fixed time series from 'renewables' table
    for i, re in nd['renewables'].iterrows():
        if re['active']:
            # set static outflow values
            outflow_args = {'nominal_value': re['capacity'],
                            'fixed': True}
            # get time series for node and parameter
            for col in nd['timeseries'].columns.values:
                if col.split('.')[0] == re['label']:
                    outflow_args[col.split('.')[1]] = nd['timeseries'][col]


            # create
            nodes.append(
                solph.Source(label=re['label'],
                             outputs={
                                 busd[re['to']]: solph.Flow(**outflow_args)})
            )
    

    # Create Sink objects with fixed time series from 'demand' table  -----> there is also the demand related to the electric boiler for the THERMAL sector
    for i, de in nd['demand'].iterrows():
        if de['active']:
            # set static inflow values
            inflow_args = {'nominal_value': de['nominal value'],
                           'fixed': de['fixed']}
            # get time series for node and parameter
            for col in nd['timeseries'].columns.values:
                if col.split('.')[0] == de['label']:
                    inflow_args[col.split('.')[1]] = nd['timeseries'][col]

            # create
            nodes.append(
                solph.Sink(label=de['label'],
                           inputs={
                               busd[de['from']]: solph.Flow(**inflow_args)})
            )
        

    #sink related to the electric boiler ---> TLR sector     (R1)   
#    my_demand_series=nd['hp_av']['Reg_1']
#    nominal_demand=nd['hp_nv_TLR']['Reg_1'].values[0]
#    nodes.append(
#            solph.Sink(label='R1_load_th',
#                       inputs={
#                           busd['R1_bus_th']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#    
#    
#    #sink related to the electric boiler ---> TLR sector     (R2)   
#    my_demand_series=nd['hp_av']['Reg_2']
#    nominal_demand=nd['hp_nv_TLR']['Reg_2'].values[0]
#    nodes.append(
#            solph.Sink(label='R2_load_th',
#                       inputs={
#                           busd['R2_bus_th']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#        
#    #sink related to the electric boiler ---> TLR sector     (R3)   
#    my_demand_series=nd['hp_av']['Reg_3']
#    nominal_demand=nd['hp_nv_TLR']['Reg_3'].values[0]
#    nodes.append(
#            solph.Sink(label='R3_load_th',
#                       inputs={
#                           busd['R3_bus_th']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#        
#        
#    #sink related to the electric sector     (R1)   
#    my_demand_series=nd['el_av']['Reg_1']
#    nominal_demand=nd['el_nv']['Reg_1'].values[0]
#    nodes.append(
#            solph.Sink(label='R1_load_cool',
#                       inputs={
#                           busd['R1_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#    
#    
#    #sink related to the electric sector     (R2)   
#    my_demand_series=nd['el_av']['Reg_2']
#    nominal_demand=nd['el_nv']['Reg_2'].values[0]
#    nodes.append(
#            solph.Sink(label='R2_load_cool',
#                       inputs={
#                           busd['R2_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#        
#    #sink related to the electric sector     (R3)   
#    my_demand_series=nd['el_av']['Reg_3']
#    nominal_demand=nd['el_nv']['Reg_3'].values[0]
#    nodes.append(
#            solph.Sink(label='R3_load_cool',
#                       inputs={
#                           busd['R3_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#    
#    
#    #sink related to the electric sector     (R4)   
#    my_demand_series=nd['el_av']['Reg_4']
#    nominal_demand=nd['el_nv']['Reg_4'].values[0]
#    nodes.append(
#            solph.Sink(label='R4_load_cool',
#                       inputs={
#                           busd['R4_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#        
#    #sink related to the electric sector     (R5)   
#    my_demand_series=nd['el_av']['Reg_5']
#    nominal_demand=nd['el_nv']['Reg_5'].values[0]
#    nodes.append(
#            solph.Sink(label='R5_load_cool',
#                       inputs={
#                           busd['R5_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )
#    
#    
#    #sink related to the electric sector     (R6)   
#    my_demand_series=nd['el_av']['Reg_6']
#    nominal_demand=nd['el_nv']['Reg_6'].values[0]
#    nodes.append(
#            solph.Sink(label='R6_load_cool',
#                       inputs={
#                           busd['R6_bus_el']: solph.Flow(actual_value=my_demand_series, fixed=True, nominal_value=nominal_demand)})
#        )       
        
    # Create Transformer objects from 'transformers' table
    for i, t in nd['transformers_pp'].iterrows():
        if t['active']:
            # set static inflow values
            inflow_args = {'variable_costs': t['variable input costs']}
            # get time series for inflow of transformer
            for col in nd['timeseries'].columns.values:
                if col.split('.')[0] == t['label']:
                    inflow_args[col.split('.')[1]] = nd['timeseries'][col]

            # create
            nodes.append(
                solph.Transformer(
                    label=t['label'],
                    inputs={busd[t['from']]: solph.Flow(**inflow_args)},
                    outputs={busd[t['to']]: solph.Flow(
                            nominal_value=t['capacity'])},
                    conversion_factors={busd[t['to']]: t['efficiency']})
            )

    # Create Transformer objects from 'transformers' table
    for i, t in nd['transformers_chp'].iterrows():
        if t['active']:
            # set static inflow values
            inflow_args = {'variable_costs': t['variable input costs']}
            # get time series for inflow of transformer
            for col in nd['timeseries'].columns.values:
                if col.split('.')[0] == t['label']:
                    inflow_args[col.split('.')[1]] = nd['timeseries'][col]

            # create
            nodes.append(
                solph.Transformer(
                    label=t['label'],
                    inputs={busd[t['from']]: solph.Flow(**inflow_args)},
                    outputs={busd[t['to']]: solph.Flow(
                            nominal_value=t['capacity']),
                            busd[t['andto']]: solph.Flow(
                            nominal_value=t['capacity_th'])},
                    conversion_factors={busd[t['to']]: t['efficiency'],
                                        busd[t['andto']]: t['efficiency_th']})
            )
       

    for i, s in nd['storages'].iterrows():
        if s['active']:
            nodes.append(
                solph.components.GenericStorage(
                    label=s['label'],
                    inputs={busd[s['bus']]: solph.Flow(
                        nominal_value=s['capacity inflow'],
                        variable_costs=s['variable input costs'])},
                    outputs={busd[s['bus']]: solph.Flow(
                        nominal_value=s['capacity outflow'],
                        variable_costs=s['variable output costs'])},
                    nominal_capacity=s['nominal capacity'],
                    capacity_loss=s['capacity loss'],
                    initial_capacity=s['initial capacity'],
                    capacity_max=s['capacity max'],
                    capacity_min=s['capacity min'],
                    inflow_conversion_factor=s['efficiency inflow'],
                    outflow_conversion_factor=s['efficiency outflow'])
            )

    for i, p in nd['powerlines'].iterrows():
        if p['active']:
            bus1 = busd[p['bus_1']]
            bus2 = busd[p['bus_2']]
            nodes.append(
                solph.custom.Link(
                    label='powerline'
                          + '_' + p['bus_1']
                          + '_' + p['bus_2'],
                    inputs={bus1: solph.Flow(),
                            bus2: solph.Flow()},
                    outputs={bus1: solph.Flow(nominal_value=p['capacity_1']),
                             bus2: solph.Flow(nominal_value=p['capacity_2'])
                             },
                    conversion_factors={(bus1, bus2): p['efficiency'],
                                        (bus2, bus1): p['efficiency']})
            )

    return nodes

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# costs
def cost_tech(Capacity, Inv, life, OM, interest):
    Inv_cost_tech = Capacity*Inv*interest/(1-(1+interest)**(-life))
    Operational_cost_tech = Capacity*Inv*OM/100.
    return Inv_cost_tech + Operational_cost_tech

def is_nan(x):
    return isinstance(x, float) and math.isnan(x)

def inv_costs(DatiCosti, excel_nodes): #[€/MWh]
    ''' 
    dic_distr = result distribution of the dispatch optimization
    df_m = dataframe of current csv of the technologies             CHANGE???????
    df = dataframe of the cost database csv
    dic = efficiencies of power plants
    N_reg = number of regions
    Cost_NG = cost natural gas [€/MWh]
    the calculations are made in MWh
    '''
    #param=pd.ExcelFile(File)
    c1= excel_nodes["renewables"]
    c2= excel_nodes["storages"]
    c3= excel_nodes["transformers_pp"]
    c4= excel_nodes["transformers_chp"]

    df_m=pd.DataFrame()
    df_m['label']=c1['label'].append(c2['bus']).append(c3['label']).append(c4['label']).append(c2['label']).append(c2['label'])
    df_m['to']=c1['to'].append(c2['label']).append(c3['to']).append(c4['to']).append(c2['bus']).append(c2['bus'])
    df_m['capacity [MW]v[MWh]']=c1['capacity'].append(c2['nominal capacity']).append(c3['capacity']).append(c4['capacity']).append(c2['capacity inflow']).append(c2['capacity outflow'])
    df_m=df_m.reset_index()
    del df_m['index']

    df_C=pd.DataFrame()
            
    for i in range(6): 
        df_c=DatiCosti.copy()
        
        for a in range(len(df_c)):
            df_c.loc[a,'to']='R' + str(i+1) + '_' + str(df_c['to'][a])
            df_c.loc[a,'label']='R' + str(i+1) + '_' + str(df_c['label'][a])
        df_C=df_C.append(df_c)
        
        
    df_C=df_C.reset_index()
    del df_C['index']
    
    DF_final= pd.merge(df_m, df_C, how='outer', on=['label', 'to']).drop_duplicates().reset_index()
    del DF_final['index']
    #result = pd.concat([df_m, df_C], axis=1, sort=False,ignore_index=True)

    DF_final['Cost [M€]']=0    
    
    for a in range(len(DF_final)):
            if is_nan(DF_final['Investment [euro/kW]'][a]):
                pass
#                print('Nan')
            else:
                #if df_m['label'][a] in df_c['label'].tolist() and df_m['to'][a] in df_c['to'].tolist():
#                    print(df_m['source'][a])
                 #   if  not is_nan(df_m['capacity'][a]): #df_m['nominal_value'][a] !=0 and
#                        df2= df.set_index('source')
    #                    print(df2)
                        Cap=DF_final['capacity [MW]v[MWh]'][a]*1000.
#                        print(colored(df_m['label'][a], 'magenta'), colored(df_m['source'][a], 'magenta'))
#                        print(df_m['nominal_value'][a])
#                        print(Cap)
                        Inv = DF_final['Investment [euro/kW]'][a]
#                        print(Inv)
                        life= DF_final['lifetime [y]'][a]
#                        print(life)
                        OM= DF_final['O&M [%Inv]'][a]
#                        print(OM)
                        #cost_i[a]= cost_tech(Cap, Inv, life, OM, 0.03)
                        cost_i = cost_tech(Cap, Inv, life, OM, 0.03)
                        #cost_i = cost_i.tolist()
                        DF_final.loc[a,"Cost [M€]"]=cost_i/(10**6)
                        
                        
    for b in range(len(DF_final)):
            if is_nan(DF_final['Investment [euro/MWh]'][b]):
                pass
#                print('Nan')
            else:
                #if df_m['label'][a] in df_c['label'].tolist() and df_m['to'][a] in df_c['to'].tolist():
#                    print(df_m['source'][a])
                 #   if  not is_nan(df_m['capacity'][a]): #df_m['nominal_value'][a] !=0 and
#                        df2= df.set_index('source')
    #                    print(df2)
                        Cap=DF_final['capacity [MW]v[MWh]'][b]
#                        print(colored(df_m['label'][a], 'magenta'), colored(df_m['source'][a], 'magenta'))
#                        print(df_m['nominal_value'][a])
#                        print(Cap)
                        Inv = DF_final['Investment [euro/MWh]'][b]
#                        print(Inv)
                        life= DF_final['lifetime [y]'][b]
#                        print(life)
                        OM= DF_final['O&M [%Inv]'][b]
#                        print(OM)
                        #cost_i[a]= cost_tech(Cap, Inv, life, OM, 0.03)
                        cost_i = cost_tech(Cap, Inv, life, OM, 0.03)
                        #cost_i = cost_i.tolist()
                        DF_final.loc[b,"Cost [M€]"]=cost_i/(10**6)                      

    Costs=pd.DataFrame(index=['Wind','Solar', 'Geo', 'RH', 'Bio', 'storage phs', 'storage batt', 'pp_gas', 'pp_water', 'chp_gas'],columns=['cost'])   
    cwind=[]
    csolar=[]
    cGeo=[]
    cRH=[]
    cBio=[]
    cphs=[]
    cbatt=[]
    cwater=[]
    cppgas=[]
    cchpgas=[]
    
    for c in range(len(DF_final)):                         
        label1 = DF_final['label'][c].split("_")
        label2 = DF_final['to'][c].split("_")
                
        if ('wind' in label1) or ('wind' in label2): #see if one of the words in the sentence is the word we want
            cwind.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['Wind']['cost']=np.nansum(cwind)
     
        if ('solar' in label1) or ('solar' in label2): #see if one of the words in the sentence is the word we want
            csolar.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['Solar']['cost']=np.nansum(csolar)

        if ('Geo' in label1) or ('Geo' in label2): #see if one of the words in the sentence is the word we want
            cGeo.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['Geo']['cost']=np.nansum(cGeo)

        if ('RH' in label1) or ('RH' in label2): #see if one of the words in the sentence is the word we want
            cRH.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['RH']['cost']=np.nansum(cRH)
     
        if ('Bio' in label1) or ('Bio' in label2): #see if one of the words in the sentence is the word we want
            cBio.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['Bio']['cost']=np.nansum(cBio)
     
        if ('phs' in label1) or ('phs' in label2): #see if one of the words in the sentence is the word we want
            cphs.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['storage phs']['cost']=np.nansum(cphs)
 
        if ('batt' in label1) or ('batt' in label2): #see if one of the words in the sentence is the word we want
            cbatt.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['storage batt']['cost']=np.nansum(cbatt)
  
        if ('water' in label1) or ('water' in label2): #see if one of the words in the sentence is the word we want
            cwater.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['pp_water']['cost']=np.nansum(cwater)

        if ('gas' in label1) and ('chp' in label1): #see if one of the words in the sentence is the word we want
            cchpgas.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['chp_gas']['cost']=np.nansum(cchpgas)
            
        if ('gas' in label1) and ('pp' in label1): #see if one of the words in the sentence is the word we want
            cppgas.append(DF_final.loc[c]['Cost [M€]'])
            Costs.loc['pp_gas']['cost']=np.nansum(cppgas)
  
    return Costs



def costs(DatiCosti, excel_nodes,f1,f2,f3,f4,f5,f6):
    
    tech_costs=inv_costs(DatiCosti, excel_nodes)
    tech_costs=tech_costs.rename(columns={'cost':'tech'})

    el_costs=pd.DataFrame(selectric(excel_nodes,f1,f2,f3,f4,f5,f6)[0], columns=['electric'], index=['NG'])

    COST_DF= tech_costs.join(el_costs, how='outer')
    #tot_costs=pd.DataFrame(COST_DF.sum(axis=1),columns=['[M€/y]'])
    costs=round(COST_DF.sum(axis=1).sum(), 2) #[M€/y]
    
    return costs        

#------------------------------------------------------------------------------ 
# costs
def co2(DatiFuel, excel_nodes, f1,f2,f3,f4,f5,f6,f7,f8,f9):


    e1= DatiFuel.parse("Electric")
    
    c7= excel_nodes["transformers_pp"]
    c8= excel_nodes["transformers_chp"]
    # flussi
    
    
    f_R1_pp_gas=f1[(('R1_pp_gas', 'R1_bus_el'), 'flow')]
    f_R1_chp_gas=f1[(('R1_chp_gas', 'R1_bus_el'), 'flow')]
    #f_R1_boiler_coal=f1[(('R1_bus_el_shortage', 'R1_bus_el'), 'flow')]
#    f_R1_boiler_th=f7[(('R1_bus_th_shortage', 'R1_bus_th'), 'flow')]
    
    
    f_R2_pp_gas=f2[(('R2_pp_gas', 'R2_bus_el'), 'flow')]
    f_R2_chp_gas=f2[(('R2_chp_gas', 'R2_bus_el'), 'flow')]
    #f_R2_boiler_coal=f2[(('R2_bus_el_shortage', 'R2_bus_el'), 'flow')]
#    f_R2_boiler_th=f8[(('R2_bus_th_shortage', 'R2_bus_th'), 'flow')]
    
    f_R3_pp_gas=f3[(('R3_pp_gas', 'R3_bus_el'), 'flow')]
    f_R3_chp_gas=f3[(('R3_chp_gas', 'R3_bus_el'), 'flow')]
    #f_R3_boiler_coal=f3[(('R3_bus_el_shortage', 'R3_bus_el'), 'flow')]
#    f_R3_boiler_th=f9[(('R3_bus_th_shortage', 'R3_bus_th'), 'flow')]
    
    f_R4_pp_gas=f4[(('R4_pp_gas', 'R4_bus_el'), 'flow')]
    #f_R4_boiler_coal=f4[(('R4_bus_el_shortage', 'R4_bus_el'), 'flow')]
    
    f_R5_pp_gas=f5[(('R5_pp_gas', 'R5_bus_el'), 'flow')]
    #f_R5_boiler_coal=f5[(('R5_bus_el_shortage', 'R5_bus_el'), 'flow')]
    
    f_R6_pp_gas=f6[(('R6_pp_gas', 'R6_bus_el'), 'flow')]
    #f_R6_boiler_coal=f6[(('R6_bus_el_shortage', 'R6_bus_el'), 'flow')]
    
    # co2
    e1=e1.set_index('Cost-CO2')
    print(e1)
    
    e_NG=(e1.loc["CO2 [Kg/GWh]"]["NG"])/(10**6)
#    e_coal=(e1.loc["CO2 [Kg/GWh]"]["coal"])/(10**6)
    
    
    c_R1_pp_gas_eff=c7.loc[0]["efficiency"]
    c_R2_pp_gas_eff=c7.loc[2]["efficiency"]
    c_R3_pp_gas_eff=c7.loc[4]["efficiency"]
    c_R4_pp_gas_eff=c7.loc[6]["efficiency"]
    c_R5_pp_gas_eff=c7.loc[8]["efficiency"]
    c_R6_pp_gas_eff=c7.loc[10]["efficiency"]
    
    c_R1_chp_gas_eff=c8.loc[0]["efficiency"]
    c_R2_chp_gas_eff=c8.loc[1]["efficiency"]
    c_R3_chp_gas_eff=c8.loc[2]["efficiency"]
    
    
    
    R1_gas=(f_R1_pp_gas*(e_NG)/c_R1_pp_gas_eff) #+R1_hard_coal
    R2_gas=(f_R2_pp_gas*(e_NG)/c_R2_pp_gas_eff) #+R2_hard_coal
    R3_gas=(f_R3_pp_gas*(e_NG)/c_R3_pp_gas_eff) #+R1_hard_coal
    R4_gas=(f_R4_pp_gas*(e_NG)/c_R4_pp_gas_eff) #+R2_hard_coal
    R5_gas=(f_R5_pp_gas*(e_NG)/c_R5_pp_gas_eff) #+R1_hard_coal
    R6_gas=(f_R6_pp_gas*(e_NG)/c_R6_pp_gas_eff) #+R2_hard_coal
    #GAS=(R1_gas,R2_gas,R3_gas,R4_gas,R5_gas,R6_gas)
    
    R1_chp_gas=(f_R1_chp_gas*(e_NG)/c_R1_chp_gas_eff)
    R2_chp_gas=(f_R2_chp_gas*(e_NG)/c_R2_chp_gas_eff)
    R3_chp_gas=(f_R3_chp_gas*(e_NG)/c_R3_chp_gas_eff)
    
   
    #preparing plot
    R1_el=R1_gas+R1_chp_gas
    R2_el=R2_gas+R2_chp_gas
    R3_el=R3_gas+R3_chp_gas
    R4_el=R4_gas
    R5_el=R5_gas
    R6_el=R6_gas
    
    REL_save=(R1_el,R2_el,R3_el,R4_el,R5_el,R6_el)
        
    
    columns=['Electric s.']
    index=['R1','R2','R3','R4','R5','R6']
    CO2={'Electric s.': REL_save}
    CO2=pd.DataFrame(CO2, index=index, columns=columns).T
    #CO2['Total [MtonsCO2/year]']=CO2.sum(axis=1).sum()/(10**6)
    co2=round(CO2.sum(axis=1).sum()/(10**6),2)
    return co2