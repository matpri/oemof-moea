# oemof-moea
Coupling of the oemof framework to a Multi-Objective Evolutionary Algorithm

Requirements
- Oemof==0.2.3
- Gurobi or other solvers (cbc, glpk..)

How to run the example of the Italian energy system (with 8 processes in parallel):
- open the Anaconda command prompt
- T:
- cd path_of_your_directory
- python Simulation_function.py

Input of the model:
- scenario2015.xlsx, the baseline of the Italian energy system 2015 in the Oemof input file (of the excel_reader example)
- Dati_Costi.xlsx
- DatiFuel.xlsx

Output of the model:
- SCENARIOS.xlsx where each row is a different evaluated scenario and the last two columns are the chosen objectives of the optimization
