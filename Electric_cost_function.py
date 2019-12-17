# -*- coding: utf-8 -*-
"""
@author: VCasalicchio
"""

"""
General description
-------------------
Il programma dà in output i seguenti elementi:
    
costgas ---> il costo totale del gas per il settore elettrico [M€]
PPGAS ---> il costo del gas di pp per regione [€]
CHPGAS ---> il costo del gas di chp per regione [€]

Data
----
I dati richiesti dalla funzione sono i seguenti
"ITALIA_2015.xlsx" 
"""

from termcolor import colored

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import pylab
from matplotlib import pylab

#print(os.path.join(os.path.dirname(__file__))) 

#File=os.path.join(os.path.dirname(__file__), 'RESULTS_dispatch.xlsx',)
#param=pd.ExcelFile(File)

def selectric(excel_nodes,f1,f2,f3,f4,f5,f6):
    
    c1= excel_nodes["buses"]
    c2= excel_nodes["commodity_sources"]
    c3= excel_nodes["demand"]
    c4= excel_nodes["powerlines"]
    c5= excel_nodes["renewables"]
    c6= excel_nodes["storages"]
    c7= excel_nodes["transformers_pp"]
    c8= excel_nodes["transformers_chp"]
    

    # flussi
    type(f2)
    f_R1_storage=f1[(('R1_storage_batt', 'R1_bus_el'), 'flow')]
    f_R1_excess=f1[(('R1_bus_el', 'R1_bus_el_excess'), 'flow')]
    f_R1_pp_gas=f1[(('R1_pp_gas', 'R1_bus_el'), 'flow')]
    f_R1_chp_gas=f1[(('R1_chp_gas', 'R1_bus_el'), 'flow')]
    
    f_R2_storage=f2[(('R2_storage_batt', 'R2_bus_el'), 'flow')]
    f_R2_excess=f2[(('R2_bus_el', 'R2_bus_el_excess'), 'flow')]
    f_R2_pp_gas=f2[(('R2_pp_gas', 'R2_bus_el'), 'flow')]
    f_R2_chp_gas=f2[(('R2_chp_gas', 'R2_bus_el'), 'flow')]
    
    f_R3_storage=f3[(('R3_storage_batt', 'R3_bus_el'), 'flow')]
    f_R3_excess=f3[(('R3_bus_el', 'R3_bus_el_excess'), 'flow')]
    f_R3_pp_gas=f3[(('R3_pp_gas', 'R3_bus_el'), 'flow')]
    f_R3_chp_gas=f3[(('R3_chp_gas', 'R3_bus_el'), 'flow')]
    
    f_R4_storage=f4[(('R4_storage_batt', 'R4_bus_el'), 'flow')]
    f_R4_excess=f4[(('R4_bus_el', 'R4_bus_el_excess'), 'flow')]
    f_R4_pp_gas=f4[(('R4_pp_gas', 'R4_bus_el'), 'flow')]
    
    f_R5_storage=f5[(('R5_storage_batt', 'R5_bus_el'), 'flow')]
    f_R5_excess=f5[(('R5_bus_el', 'R5_bus_el_excess'), 'flow')]
    f_R5_pp_gas=f5[(('R5_pp_gas', 'R5_bus_el'), 'flow')]
    
    f_R6_storage=f6[(('R6_storage_batt', 'R6_bus_el'), 'flow')]
    f_R6_excess=f6[(('R6_bus_el', 'R6_bus_el_excess'), 'flow')]
    f_R6_pp_gas=f6[(('R6_pp_gas', 'R6_bus_el'), 'flow')]

    
    
    
    # costi e efficienze
    
    c_R1_storage=c6.loc[1]["variable output costs"]
    c_R2_storage=c6.loc[3]["variable output costs"]
    c_R3_storage=c6.loc[5]["variable output costs"]
    c_R4_storage=c6.loc[7]["variable output costs"]
    c_R5_storage=c6.loc[9]["variable output costs"]
    c_R6_storage=c6.loc[11]["variable output costs"]
        
    """
    c_R1_shortage=c1.loc[7]["shortage costs"]
    c_R2_shortage=c1.loc[8]["shortage costs"]
    
    c_uranium=c2.loc[0]["variable costs"]
    c_R1_pp_uranium_t=c7.loc[0]["variable input costs"]
    c_R2_pp_uranium_t=c7.loc[7]["variable input costs"]
    c_R1_pp_uranium_eff=c7.loc[0]["efficiency"]
    c_R2_pp_uranium_eff=c7.loc[7]["efficiency"]
    
    c_lignite=c2.loc[1]["variable costs"]
    c_R1_pp_lignite_t=c7.loc[1]["variable input costs"]
    c_R2_pp_lignite_t=c7.loc[8]["variable input costs"]
    c_R1_pp_lignite_eff=c7.loc[1]["efficiency"]
    c_R2_pp_lignite_eff=c7.loc[8]["efficiency"]
    
    c_hard_coal=c2.loc[2]["variable costs"]
    c_R1_pp_hard_coal_t=c7.loc[2]["variable input costs"]
    c_R2_pp_hard_coal_t=c7.loc[9]["variable input costs"]
    c_R1_pp_hard_coal_eff=c7.loc[2]["efficiency"]
    c_R2_pp_hard_coal_eff=c7.loc[9]["efficiency"]
    """
    c_gas=c2.loc[0]["variable costs"]
    
    c_R1_pp_gas_t=24.912
    c_R2_pp_gas_t=24.912
    c_R3_pp_gas_t=24.912
    c_R4_pp_gas_t=24.912
    c_R5_pp_gas_t=24.912
    c_R6_pp_gas_t=24.912
    c_R1_pp_gas_eff=c7.loc[0]["efficiency"]
    c_R2_pp_gas_eff=c7.loc[2]["efficiency"]
    c_R3_pp_gas_eff=c7.loc[4]["efficiency"]
    c_R4_pp_gas_eff=c7.loc[6]["efficiency"]
    c_R5_pp_gas_eff=c7.loc[8]["efficiency"]
    c_R6_pp_gas_eff=c7.loc[10]["efficiency"]
    
    c_R1_chp_gas_t=24.912
    c_R2_chp_gas_t=24.912
    c_R3_chp_gas_t=24.912
    c_R1_chp_gas_eff=c8.loc[0]["efficiency"]
    c_R2_chp_gas_eff=c8.loc[1]["efficiency"]
    c_R3_chp_gas_eff=c8.loc[2]["efficiency"]
    
    
    R1_gas=(f_R1_pp_gas*(c_gas+c_R1_pp_gas_t)/c_R1_pp_gas_eff) #+R1_hard_coal
    R2_gas=(f_R2_pp_gas*(c_gas+c_R2_pp_gas_t)/c_R2_pp_gas_eff) #+R2_hard_coal
    R3_gas=(f_R3_pp_gas*(c_gas+c_R3_pp_gas_t)/c_R3_pp_gas_eff) #+R1_hard_coal
    R4_gas=(f_R4_pp_gas*(c_gas+c_R4_pp_gas_t)/c_R4_pp_gas_eff) #+R2_hard_coal
    R5_gas=(f_R5_pp_gas*(c_gas+c_R5_pp_gas_t)/c_R5_pp_gas_eff) #+R1_hard_coal
    R6_gas=(f_R6_pp_gas*(c_gas+c_R6_pp_gas_t)/c_R6_pp_gas_eff) #+R2_hard_coal
    GAS=(R1_gas,R2_gas,R3_gas,R4_gas,R5_gas,R6_gas)
    

    
    R1_chp=(f_R1_chp_gas*(c_gas+c_R1_chp_gas_t)/c_R1_chp_gas_eff)
    R2_chp=(f_R2_chp_gas*(c_gas+c_R2_chp_gas_t)/c_R2_chp_gas_eff)
    R3_chp=(f_R3_chp_gas*(c_gas+c_R3_chp_gas_t)/c_R3_chp_gas_eff)
    """
    R1_chp_gas=R1_chp+R1_gas
    R2_chp_gas=R2_chp+R2_gas
    R3_chp_gas=R3_chp+R3_gas

    CHP_GAS=(R1_chp_gas,R2_chp_gas,R3_chp_gas,0,0,0)
    """
    PPGAS=[R1_gas,R2_gas,R3_gas,R4_gas,R5_gas,R6_gas]
    CHPGAS=[R1_chp,R2_chp,R3_chp,0,0,0]
    
    costgas=(sum(PPGAS)+sum(CHPGAS))/(10**6)
    
    return costgas, PPGAS, CHPGAS
    
    PPGAS=selectric()[0]
    CHPGAS=selectric()[1]
    
    
    
    
"""    

    GAS=(PPGAS[0],PPGAS[1],PPGAS[2],PPGAS[3],PPGAS[4],PPGAS[5])
    CHP_GAS=(CHPGAS[0]+PPGAS[0],CHPGAS[1]+PPGAS[1],CHPGAS[2]+PPGAS[2],CHPGAS[3]+PPGAS[3],CHPGAS[4]+PPGAS[4],CHPGAS[5]+PPGAS[5])

    R1_biomass=(f_R1_pp_biomass*(c_biomass+c_R1_pp_biomass_t)/c_R1_pp_biomass_eff)+R1_gas
    R2_biomass=(f_R2_pp_biomass*(c_biomass+c_R2_pp_biomass_t)/c_R2_pp_biomass_eff)+R2_gas
    BIOMASS=(R1_biomass,R2_biomass)


    R1_storage_batt=(f_R1_storage*c_R1_storage)+R1_chp_gas
    R2_storage_batt=(f_R2_storage*c_R2_storage)+R2_chp_gas
    R3_storage_batt=(f_R3_storage*c_R3_storage)+R3_chp_gas
    R4_storage_batt=(f_R4_storage*c_R4_storage)+R4_gas
    R5_storage_batt=(f_R5_storage*c_R5_storage)+R5_gas
    R6_storage_batt=(f_R6_storage*c_R6_storage)+R6_gas
    STORAGE_BATT=(R1_storage_batt,R2_storage_batt,R3_storage_batt,R4_storage_batt,R5_storage_batt,R6_storage_batt)

    
    #print('..................')
    #print(GeoList, BioListt, RHListt, PVListt, WListt, PPListt)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(19,14))
    N = len(GAS)
    ind = np.arange(N)
    width = 0.6
    
    #red_line,= axes.plot(ind, loadlist, marker='o', color='red', linewidth=1.5, label='Demand')
    #p1 = axes.bar(ind,  STORAGE_BATT, width, color='silver', linewidth=0.5, edgecolor='black')
    #p1 = axes.bar(ind, BIOMASS, width, color='y', linewidth=0.5, edgecolor='black')
    p2 = axes.bar(ind, CHP_GAS, width, color='y', linewidth=0.5, edgecolor='black')
    p3 = axes.bar(ind, GAS, width, color='goldenrod', linewidth=0.5, edgecolor='black')
    #p3 = axes.bar(ind, HARD_COAL, width, color='saddlebrown', linewidth=0.5, edgecolor='black')
    #p4 = axes.bar(ind, LIGNITE, width, color='peru', linewidth=0.5, edgecolor='black')
    #p5 = axes.bar(ind, MIXED_FUELS, width, color='tomato', linewidth=0.5, edgecolor='black')
    #p6 = axes.bar(ind, OIL, width, color='firebrick', linewidth=0.5, edgecolor='black')
    #p7 = axes.bar(ind,  URANIUM, width, color='maroon', linewidth=0.5, edgecolor='black')
    #p8 = axes.bar(ind,  SHORTAGE, width, color='silver', linewidth=0.5, edgecolor='black')
    
    #p4 = axes.bar(ind,  RHListt, width, color='skyblue', linewidth=0.5, edgecolor='black')
    #p5 = axes.bar(ind,  BioListt, width, color='forestgreen', linewidth=0.5, edgecolor='black')
    #p6 = axes.bar(ind,  GeoListt, width, color='peru', linewidth=0.5, edgecolor='black')
    
    
    axes.set_ylabel('Costs [€]', fontsize=15)
    axes.tick_params(axis='y',labelsize=11)
    axes.set_xticklabels(['R1','R2','R3','R4','R5','R6',], fontsize=15)
    axes.yaxis.offsetText.set_fontsize(10) 
    #        axes[a].set_title('RS', fontsize=12)#+ width/2.
    #axes[a].set_ylim([0, 7000])
    #
    #axes[a].set_yticklabels([])
    #
    #axes[a].set_ylim([0, 7000])
    #axes[a].set_xticklabels(['P'+str(a),], fontsize=12)    
    #        axes[a].set_title('P'+str(a), fontsize=12)
    axes.set_xticks(ind ,)#+ width/2.
    #    axes[a].set_xticklabels(['Costs \nper \nsource',]) 
    axes.grid(linestyle='dotted')
    art = []
    handles = [p2,p3]#[mpl.patches.Rectangle((0,0), 0,0, facecolor=pol.get_facecolor()[0]) for pol in sp]
    #handles =list(handles)
    #handles=handles+(red_line,)
    lgd = plt.legend(handles,( 'storage batt','chp_gas','gas'), bbox_to_anchor=(1.01, 0.5), loc='center left', prop={'size': 14}, ncol=1)
    #lgd = plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]),('Power Plants', 'Wind', 'PV', 'River Hydro','Biomass PP', 'Geothermal'), bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1)
    art.append(lgd)
    pylab.savefig("Costs.png", additional_artists=art, bbox_inches="tight", dpi=300)
    plt.show()
"""