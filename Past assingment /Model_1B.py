from gurobipy import *
from math import *
import numpy as np
import pandas as pd
import timeit
from xlwt import Workbook

start = timeit.default_timer()

#######################################################################################################################
##############################################   DATA   ###############################################################
#######################################################################################################################

num_ac_types = 3 # Number of aircraft types
LF = 0.8 # Load factor
BT = 10 * 7 # Block time [h/week] - average utilisation time (same for all aircraft types k)
f = 1.42 # Fuel cost [USD/gallon]

# Extracting demand from Excel file 'results.xlsx'
xls = pd.ExcelFile('results.xlsx')
data_demand = pd.read_excel(xls, 'Forecast demand')
q = np.array(data_demand)[::,1::]  # Demand (matrix) between airports i and j
demand = q

# Airports
Airports = ['ESGG', 'ESMS', 'ESPA', 'ESSA', 'ESFR', 'ESDF', 'ESIB', 'SE-0016', 'ESNS', 'ESGR', 'ESNY', 'ESKN', 'ESNB', 'ESCM', 'ESGO']
airports = range(len(Airports))

runway_lengths = [3299, 2800, 3350, 3301, 1987, 2331, 2264, 2500, 2520, 1736, 2524, 2878, 820, 1963, 890]

# Creating matrix with max runway length between origin and destination
runways = np.zeros((len(runway_lengths), len(runway_lengths)))
for origin in range(len(runway_lengths)):
    for destination in range(len(runway_lengths)):
        if origin != destination:
            if runway_lengths[origin] < runway_lengths[destination]:
                runways[origin][destination] = runway_lengths[origin]
            else:
                runways[origin][destination] = runway_lengths[destination]

# Extracting the distance between airports i and j from Excel file 'results.xlsx'
data_distances = pd.read_excel(xls, 'Distances')
dist = np.array(data_distances)[::,1::]  # Distances (matrix) between airports i and j

# Aircraft parameters
Aircraft = {
    'speed': [550, 820, 850], # Speed of the aircraft type k - spk [km/h]
    'seats': [45, 70, 150], # Number of seats per aircraft type k - sk [-]
    'TAT': [25, 35, 45], # Average turnaround time [min]
    'range': [1500, 3300, 6300], # Maximum range [km]
    'runway': [1400, 1600, 1800], # Runway distance required [m]
    'lease': [15000, 34000, 80000], # Weekly lease cost [€]
    'op_cost': [300, 600, 1250], # Fixed operating cost [€]
    'time_cost_param': [750, 775, 1400], # Time cost parameter C_T [€/h]
    'fuel_cost_param': [1, 2, 3.75] # Fuel cost parameter C_F [-]
}

# Convert TAT from [min] to [h]
for k in range(len(Aircraft['TAT'])):
    Aircraft['TAT'][k] = Aircraft['TAT'][k] / 60

# Yield between airports i and j - Revenue per RPK flown
Yield = np.zeros((len(Airports),len(Airports)))
for i in range(len(dist)):
    for j in range(len(dist)):
        if i != j:
            Yield[i][j] = 5.9 * dist[i][j] ** (-0.76) + 0.043

# Range matrix
a = {}
for k in range(num_ac_types):
    for i in range(len(dist)):
        for j in range(len(dist[i])):
            a[i,j,k] = 0
            if dist[i][j] <= Aircraft['range'][k] :
                a[i,j,k] = 10000

# Runway matrix
run = {}
for k in range(num_ac_types):
    for i in range(len(runways)):
        for j in range(len(runways[i])):
            run[i,j,k] = 0
            if runways[i][j] >= Aircraft['runway'][k] :
                run[i,j,k] = 10000

#######################################################################################################################
##############################################   INITIALISING MODEL   #################################################
#######################################################################################################################

m = Model('Problem 1B')

# Decision variables
x = {} # direct pax flow from airport i to airport j
w = {} # pax flow from airport i to airport j with transfer at hub
z = {} # number of flights (frequency) from airport i to airport j
AC = {} # number of aircraft of type k

# Setting hub in Göteborg
g = np.ones(15) # 15 airports
g[0] = 0 # Stockholm (ESGG) is the hub

# Creating dictionaries of hub & spokes
hub = {}
spokes = {}
for j in range(len(demand)):
    if g[j] == 0: # Hub is the first one in the list
        hub[j] = j
    else:
        spokes[j] = j

for i in airports:
    for j in airports:
        x[i,j] = m.addVar(obj = Yield[i][j] * dist[i][j], lb=0, vtype=GRB.INTEGER)
        w[i,j] = m.addVar(obj = 0.9 * Yield[i][j] * dist[i][j], lb=0, vtype=GRB.INTEGER) # 10% lower revenue in case of transfer at hub

        for k in range(num_ac_types):
            C_O_ac = Aircraft['op_cost'][k]
            C_T_ac = Aircraft['time_cost_param'][k] * dist[i][j] / Aircraft['speed'][k]
            C_F_ac = Aircraft['fuel_cost_param'][k] * f * dist[i][j] / 1.5

            if g[i] == 0 or g[j] == 0: # if origin/destination is the hub
                z[i,j,k] = m.addVar(obj= -0.7 * (C_O_ac + C_T_ac + C_F_ac), lb=0, vtype=GRB.INTEGER) # 30% lower operating costs when hub is origin/destination
            else:
                z[i,j,k] = m.addVar(obj= - (C_O_ac + C_T_ac + C_F_ac), lb=0, vtype=GRB.INTEGER)

for k in range(num_ac_types):
    AC[k] =  m.addVar(obj= - (Aircraft['lease'][k]), lb=0, vtype=GRB.INTEGER)

m.update()
m.setObjective(m.getObjective(), GRB.MAXIMIZE)

#######################################################################################################################
##############################################   CONSTRAINTS   ########################################################
#######################################################################################################################

for i in airports:
    for j in airports:
        # C1: Demand constraint
        m.addConstr(x[i,j] + w[i,j] <= q[i][j])

        # C1*: Transfer pax constraint (also demand-related)
        m.addConstr(w[i,j] <= q[i][j] * g[i] * g[j])

        # C2: Capacity constraint
        m.addConstr(x[i,j] + quicksum(w[i,m] for m in range(len(demand))) * (1 - g[j]) + quicksum(w[m,j] for m in range(len(demand))) * (1 - g[i])
                    <= quicksum(z[i,j,k] * floor(Aircraft['seats'][k] * LF) for k in range(num_ac_types)))

        # C5: Range constraint
        for k in range(num_ac_types):
            m.addConstr(z[i,j,k] <= a[i,j,k])

        # "C6": Runway length constraint (new!)
        for k in range(num_ac_types):
            m.addConstr(z[i,j,k] <= run[i,j,k])

    # C3: In-out constraint
    for k in range(num_ac_types):
        m.addConstr(quicksum(z[i,j,k] for j in range(len(demand[i]))) == quicksum(z[j,i,k] for j in range(len(demand[i]))))

# C4: Time constraint
for k in range(num_ac_types):
    m.addConstr(quicksum(quicksum((dist[i][j] / Aircraft['speed'][k] + 1.5 * Aircraft['TAT'][k]) * z[i,j,k] for j in hub) for i in range(len(demand)))
                + quicksum(quicksum((dist[i][j] / Aircraft['speed'][k] + Aircraft['TAT'][k]) * z[i,j,k] for j in spokes) for i in range(len(demand)))
                <= BT * AC[k])

m.update()

print("Optimizing: ")
m.optimize()
m.write("testout.sol")
print("Done? ")
status = m.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

stop = timeit.default_timer()
print('Time: ', stop - start)

# Printing out solutions
print()
print("Frequencies:----------------------------------")
print()
for i in airports:
    for j in airports:
        for k in range(num_ac_types):
            if z[i,j,k].X >0:
                print(Airports[i], ' to ', Airports[j], z[i,j,k].X)

#######################################################################################################################
##############################################   WRITING RESULTS TO EXCEL   ###########################################
#######################################################################################################################

wb = Workbook()
overview = wb.add_sheet('Overview', cell_overwrite_ok = True)
sheet_x = wb.add_sheet('x', cell_overwrite_ok = True)
sheet_w = wb.add_sheet('w', cell_overwrite_ok = True)
for k in range(num_ac_types):
    locals()['sheet_{0}'.format(k)] = wb.add_sheet('AC %s' %(k+1), cell_overwrite_ok = True)

# Writing overview
overview.write(0, 0, "Overview")
overview.write(1, 0, "profit [EUR/week]")
overview.write(2, 0, "gap [%]")
overview.write(1, 1, m.objVal)
overview.write(2, 1, m.MIPgap * 100)

for k in range(num_ac_types):
    overview.write(3 + k, 0, 'Aircraft %s' % str(k+1))
    overview.write(3 + k, 1, AC[k].x)

# Writing results in the Excel file 'Model_1B_results.xls'
sheet_x.write(0, 0, 'x[i,j]')
sheet_w.write(0, 0, 'w[i,j]')

for k in range(num_ac_types):
    locals()['sheet_{0}'.format(k)].write(0, 0, 'aircraft %s' %(k+1))

for i in range(len(demand)):
    sheet_x.write(i + 1, 0, Airports[i])
    sheet_w.write(i + 1, 0, Airports[i])

    for k in range(num_ac_types):
        locals()['sheet_{0}'.format(k)].write(i+1, 0, Airports[i])

    for j in range(len(demand[i])):
        sheet_x.write(0, j + 1, Airports[j])
        sheet_w.write(0, j + 1, Airports[j])

        sheet_x.write(i + 1, j + 1, x[i, j].x)
        sheet_w.write(i + 1, j + 1, w[i, j].x)

        for k in range(num_ac_types):
            locals()['sheet_{0}'.format(k)].write(0, j+1, Airports[j])
            locals()['sheet_{0}'.format(k)].write(i+1, j+1, z[i,j,k].x)

# Saving the Excel file
wb.save('Model_1B_results.xls')