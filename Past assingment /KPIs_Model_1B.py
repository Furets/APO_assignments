import numpy as np
import pandas as pd

num_ac_types = 3
f = 1.42

# Extracting the results from Network_fleet_development excel file
xls1 = pd.ExcelFile('Model_1B_results.xls')
xls2 = pd.ExcelFile('results.xlsx')

# List of airports
Airports = ['ESGG', 'ESMS', 'ESPA', 'ESSA', 'ESFR', 'ESDF', 'ESIB', 'SE-0016', 'ESNS', 'ESGR', 'ESNY', 'ESKN',
                  'ESNB', 'ESCM', 'ESGO']
airports = range(len(Airports))

# Extracting distance between airport i and airport j
data_distances = pd.read_excel(xls2, 'Distances')
dist = np.array(data_distances)[::, 1::]  # Distances (matrix) between airports i and j

# Characteristics of aircraft
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

# Costs from origin i to destination j per aircraft type k
C_O = [[], [], []] # Total fixed operational costs [€]
C_T = [[], [], []] # Time-based costs [€/h]
C_F = [[], [], []] # Fuel costs [USD]
num_ac_types = 3
f = 1.42
for ac in range(num_ac_types):
    for origin in Airports:
        for destination in Airports:
            # if origin != destination:
            if origin == 'ESGG' or destination == 'ESGG':
                C_O_ac = 0.7 * Aircraft['op_cost'][ac]
                C_T_ac = 0.7 * Aircraft['time_cost_param'][ac] * dist[Airports.index(origin)][Airports.index(destination)] / Aircraft['speed'][ac]
                C_F_ac = 0.7 * Aircraft['fuel_cost_param'][ac] * f * dist[Airports.index(origin)][Airports.index(destination)] / 1.5
            else:
                C_O_ac = Aircraft['op_cost'][ac]
                C_T_ac = Aircraft['time_cost_param'][ac] * dist[Airports.index(origin)][Airports.index(destination)] / Aircraft['speed'][ac]
                C_F_ac = Aircraft['fuel_cost_param'][ac] * f * dist[Airports.index(origin)][Airports.index(destination)] / 1.5
            if ac == 0:
                C_O[ac].append(C_O_ac)
                C_T[ac].append(C_T_ac)
                C_F[ac].append(C_F_ac)
            elif ac == 1:
                C_O[ac].append(C_O_ac)
                C_T[ac].append(C_T_ac)
                C_F[ac].append(C_F_ac)
            else:
                C_O[ac].append(C_O_ac)
                C_T[ac].append(C_T_ac)
                C_F[ac].append(C_F_ac)

Cost = [[], [], []] # Total costs
for ac in range(num_ac_types):
    for flight in range(len(C_O[ac])):
        C_total = C_O[ac][flight] + C_T[ac][flight] + C_F[ac][flight] # all flights operate through the hub - 30% lower operating costs
        if ac == 0:
            Cost[ac].append(C_total)
        elif ac == 1:
            Cost[ac].append(C_total)
        else:
            Cost[ac].append(C_total)

lst1 = Cost[0]
cost_type1 = []
while lst1 != []:
    cost_type1.append(lst1[:15])
    lst1 = lst1[15:]

lst1 = Cost[0]
cost_type2 = []
while lst1 != []:
    cost_type2.append(lst1[:15])
    lst1 = lst1[15:]

lst1 = Cost[0]
cost_type3 = []
while lst1 != []:
    cost_type3.append(lst1[:15])
    lst1 = lst1[15:]

Yield = np.zeros((len(Airports),len(Airports)))
for i in Airports:
    for j in Airports:
        if i != j:
            if i == 'ESGG' or j == 'ESGG': # if origin/destination is the hub
                for k in range(len(dist)):
                    for l in range(len(dist)):
                        if k != l:
                            Yield[k][l] = 5.9 * dist[k][l] ** (-0.76) + 0.043
            else:
                for k in range(len(dist)):
                    for l in range(len(dist)):
                        if k != l:
                            Yield[k][l] = 0.9 * (5.9 * dist[k][l] ** (-0.76) + 0.043)

# Profit determined by Gurobi
data_summary = pd.read_excel(xls1, 'Overview')
profit = data_summary.iloc[0, 1]
# Total cost
totalcost = 501500
# Calculating the total revenue
totalrevenue = totalcost + profit
# Number of aircraft type 1
type1 = data_summary.iloc[2, 1]
# Number of aircraft type 2
type2 = data_summary.iloc[3, 1]
# Number of aircraft type 3
type3 = data_summary.iloc[4, 1]
# Number of aircraft used in the model
nr_aircraft = type1 + type2 + type3
# print('Number of aircraft used in the model: ', nr_aircraft, 'aircraft')

# Extracting number of flights between airport i and airport j for aircraft type 1
flights_ac1 = pd.read_excel(xls1, 'AC 1')
AC1_flights = np.array(flights_ac1)[::, 1::]  # Number of flights for aircraft type 1 (matrix) between airports i and j
# print('Number of flights for aircraft type 1 (frequencies): ', AC1_flights)
# Determine the total number of flights made by aircraft type 1 per week
nr_flights1 = 0
for i in range(len(AC1_flights)):
    for j in range(len(AC1_flights)):
        if AC1_flights[i][j] != 0:
            nr_flights1 += AC1_flights[i][j]
# print('Total number of flights made by aircraft type 1 per week:', nr_flights1)
if nr_flights1 != 0:
    # print('The average of number of flights mabe by an aircraft type 1 per week:', nr_flights1/type1)
    # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
    dicti_flights1 = {}
    for i in range(len(AC1_flights)):
        for j in range(len(AC1_flights)):
            x_routes = AC1_flights[i][j]
            if x_routes != 0:
                if Airports[i] not in dicti_flights1:
                    dicti_flights1[Airports[i]] = []
                dicti_flights1[Airports[i]].append(Airports[j])
    # print("Flights between airports:", dicti_flights1)
    # Determine the most frequented flight for aircraft type 1
    max = 0
    for i in range(len(AC1_flights)):
        for j in range(len(AC1_flights)):
            if AC1_flights[i][j] > max:
                max = AC1_flights[i][j]
                freq_aiports1 = [Airports[i], Airports[j]]
    # print('The most frequented flight for aircraft type 1 is:', freq_aiports1)
else:
    print('No flights with aircraft type 1')

# Extracting number of flights between airport i and airport j for aircraft type 2
flights_ac2 = pd.read_excel(xls1, 'AC 2')
AC2_flights = np.array(flights_ac2)[::, 1::]  # Number of flights for aircraft type 2 (matrix) between airports i and j
# print('Number of flights for aircraft type 2 (frequencies): ', AC2_flights)
# Determine the total number of flights made by aircraft type 2 per week
nr_flights2 = 0
for i in range(len(AC2_flights)):
    for j in range(len(AC2_flights)):
        if AC2_flights[i][j] != 0:
            nr_flights2 += AC2_flights[i][j]
# print('Total number of flights made by aircraft type 2 per week:', nr_flights2)
if nr_flights2 != 0:
    # print('The average of number of flights mabe by an aircraft type 2 per week:', nr_flights2/type2)
    # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
    dicti_flights2 = {}
    for i in range(len(AC2_flights)):
        for j in range(len(AC2_flights)):
            x_routes = AC2_flights[i][j]
            if x_routes != 0:
                if Airports[i] not in dicti_flights2:
                    dicti_flights2[Airports[i]] = []
                dicti_flights2[Airports[i]].append(Airports[j])
    # print("Flights between airports:", dicti_flights2)
    # Determine the most frequented flight for aircraft type 2
    max = 0
    for i in range(len(AC2_flights)):
        for j in range(len(AC2_flights)):
            if AC2_flights[i][j] > max:
                max = AC2_flights[i][j]
                freq_aiports2 = [Airports[i], Airports[j]]
    # print('The most frequented flight for aircraft type 2 is:', freq_aiports2)
else:
    print('No flights with aircraft type 2')

# Extracting number of flights between airport i and airport j for aircraft type 3
flights_ac3 = pd.read_excel(xls1, 'AC 3')
AC3_flights = np.array(flights_ac3)[::, 1::]  # Number of flights for aircraft type 3 (matrix) between airports i and j
# print('Number of flights for aircraft type 3 (frequencies): ', AC3_flights)
# Determine the total number of flights made by aircraft type 3 per week
nr_flights3 = 0
for i in range(len(AC3_flights)):
    for j in range(len(AC3_flights)):
        if AC3_flights[i][j] != 0:
            nr_flights3 += AC3_flights[i][j]
# print('Total of flights made by aircraft type 3 per week:', nr_flights3)
if nr_flights3 != 0:
    # print('The average of number of flights mabe by an aircraft type 3 per week:', nr_flights2/type2)
    # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
    dicti_flights3 = {}
    for i in range(len(AC3_flights)):
        for j in range(len(AC3_flights)):
            x_routes = AC3_flights[i][j]
            if x_routes != 0:
                if Airports[i] not in dicti_flights3:
                    dicti_flights3[Airports[i]] = []
                dicti_flights3[Airports[i]].append(Airports[j])
    # print("Flights between airports:", dicti_flights3)
    # Determine the most frequented flight for aircraft type 3
    max = 0
    for i in range(len(AC3_flights)):
        for j in range(len(AC3_flights)):
            if AC3_flights[i][j] > max:
                max = AC3_flights[i][j]
                freq_aiports3 = [Airports[i], Airports[j]]
    # print('The most frequented flight for aircraft type 3 is:', freq_aiports3)
else:
    print('No flights with aircraft type 3')

# Extracting direct number of pax between airport i and j
pax_direct = pd.read_excel(xls1, 'x')
xij = np.array(pax_direct)[::, 1::]
# Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
dicti_legsx = {}
for i in range(len(xij)):
    for j in range(len(xij)):
        x_routes = xij[i][j]
        if x_routes != 0:
            if Airports[i] not in dicti_legsx:
                dicti_legsx[Airports[i]] = []
            dicti_legsx[Airports[i]].append(Airports[j])
print("Direct flights between airports:", dicti_legsx)

# Extracting connecting number of pax at the hub going from airport i to airport j
pax_connecting = pd.read_excel(xls1, 'w')
wij = np.array(pax_connecting)[::, 1::]
# Determine the legs that are going throw the hub between airport i and airport j
dicti_legsw = {}
for i in range(len(wij)):
    for j in range(len(wij)):
        x_routes = wij[i][j]
        if x_routes != 0:
            if Airports[i] not in dicti_legsw:
                dicti_legsw[Airports[i]] = []
            dicti_legsw[Airports[i]].append(Airports[j])
print("Flights throw the hub between airports:", dicti_legsw)

# Number of seats per every type of aircraft
seats_type1 = 45
seats_type2 = 70
seats_type3 = 150

profittt = []
ask = []
rpk = []
rask = []
cask = []
yieldd = []
lf = []

for a1 in range(len(Airports)):
    for a2 in range(len(Airports)):
        if a1 != a2:
            if wij[a1][a2] == 0:
                if xij[a1][a2] == 0:
                    print("No flight between", Airports[a1], 'and', Airports[a2])
                else:
                    print('Direct flight between', Airports[a1], 'and', Airports[a2])
                    dist1 = dist[a1][a2]
                    seat1 = AC1_flights[a1][a2] * seats_type1 + AC2_flights[a1][a2] * seats_type2 + AC3_flights[a1][a2] * seats_type3
                    pax1 = xij[a1][a2]
                    cost1 = cost_type1[a1][a2] * AC1_flights[a1][a2] + cost_type2[a1][a2] * AC2_flights[a1][a2] + cost_type3[a1][a2] * AC3_flights[a1][a2]
                    revenue1 = Yield[a1][a2] * dist1 * xij[a1][a2]
                    ASK1 = dist1 * seat1
                    ask.append(ASK1)
                    RPK1 = dist1 * pax1
                    rpk.append(RPK1)
                    if ASK1 != 0:
                        RASK1 = revenue1/ASK1
                        CASK1 = cost1/ASK1
                    rask.append(RASK1)
                    cask.append(CASK1)
                    if RPK1 != 0:
                        Yield1 = revenue1/RPK1
                    yieldd.append(Yield1)
                    op1 = (RASK1 - CASK1) * ASK1
                    profittt.append(revenue1)
                    LF1 = RPK1/ASK1
                    lf.append(LF1)
                    print('ASK:', ASK1)
                    print('RPK:', RPK1)
                    print('RASK:', RASK1)
                    print('CASK:', CASK1)
                    print('Yield:', Yield1)
                    print('Unit profit:', RASK1 - CASK1)
                    print('Operational profit: ', op1)
                    print('Load factor:', LF1)
            else:
                if xij[a1][a2] != 0:
                    print("Direct and throw the hub flight between", Airports[a1], 'and', Airports[a2])
                    dist1 = dist[a1][a2]
                    seat1 = AC1_flights[a1][a2] * seats_type1 + AC2_flights[a1][a2] * seats_type2 + AC3_flights[a1][
                        a2] * seats_type3
                    pax1 = xij[a1][a2]
                    cost1 = cost_type1[a1][a2] * AC1_flights[a1][a2] + cost_type2[a1][a2] * AC2_flights[a1][a2] + \
                            cost_type3[a1][a2] * AC3_flights[a1][a2]
                    revenue1 = Yield[a1][a2] * dist1 * xij[a1][a2]
                    ASK1 = dist1 * seat1
                    RPK1 = dist1 * pax1
                    if ASK1 != 0:
                        RASK1 = revenue1 / ASK1
                        CASK1 = cost1 / ASK1
                    if RPK1 != 0:
                        Yield1 = revenue1 / RPK1
                    unit1 = RASK1 - CASK1
                    op1 = (RASK1 - CASK1) * ASK1
                    LF1 = RPK1 / ASK1

                    dist2 = dist[0][a2] + dist[a1][0]
                    seat2 = (AC1_flights[a1][0] * seats_type1 + AC2_flights[a1][0] * seats_type2 + AC3_flights[a1][
                        0] * seats_type3) + (AC1_flights[0][a2] * seats_type1 + AC2_flights[0][a2] * seats_type2 +
                                             AC3_flights[0][
                                                 a2] * seats_type3)
                    pax2 = xij[0][a2] + xij[a1][0] + wij[a1][a2]
                    cost2 = (cost_type1[0][a2] * AC1_flights[0][a2] + cost_type2[0][a2] * AC2_flights[0][a2] + \
                             cost_type3[0][a2] * AC3_flights[0][a2]) + (cost_type1[a1][0] * AC1_flights[a1][0] + \
                                                                        cost_type2[a1][0] * AC2_flights[a1][0] +
                                                                        cost_type3[a1][0] * AC3_flights[a1][0])
                    revenue2 = Yield[a1][a2] * dist1 * pax2
                    # revenue2 = (Yield[0][a2] * dist1 * xij[0][a2]) + (Yield[a1][0] * dist1 * xij[a1][0])
                    ASK2 = dist2 * seat2
                    RPK2 = dist2 * pax2 # (dist[a1][0] * xij[a1][0]) + (dist[0][a2] * xij[0][a2]) + 2 * (dist[a1][a2] * wij[a1][a2])
                    if ASK2 != 0:
                        RASK2 = revenue2 / ASK2
                        CASK2 = cost2 / ASK2
                    if RPK2 != 0:
                        Yield2 = revenue2 / RPK2
                    unit2 = RASK2 - CASK2
                    op2 = (RASK2 - CASK2) * ASK2

                    LF2 = RPK2 / ASK2

                    ASK = (ASK1 + ASK2) / 2
                    ask.append(ASK)
                    RPK = (RPK1 + RPK2) / 2
                    rpk.append(RPK)
                    RASK = (RASK1 + RASK2) / 2
                    CASK = (CASK1 + CASK2) / 2
                    rask.append(RASK)
                    cask.append(CASK)
                    Yieldt = (Yield1 + Yield2) / 2
                    yieldd.append(Yieldt)
                    unit = (unit2 + unit1) / 2
                    op = (op1 + op2) / 2
                    profittt.append(revenue1)
                    profittt.append(revenue2)
                    LF = (LF1 + LF2) / 2
                    lf.append(LF)

                    print('ASK:', ASK)
                    print('RPK:', RPK)
                    print('RASK:', RASK)
                    print('CASK:', CASK)
                    print('Yield:', Yieldt)
                    print('Unit profit:', unit)
                    print('Operational profit: ', op)
                    print('Load factor:', LF)
                else:
                    print('Throw the hub', Airports[a1], '-', 'ESGG', '-', Airports[a2])
                    dist1 = dist[0][a2] + dist[a1][0]
                    seat1 = (AC1_flights[a1][0] * seats_type1 + AC2_flights[a1][0] * seats_type2 + AC3_flights[a1][
                        0] * seats_type3) + (AC1_flights[0][a2] * seats_type1 + AC2_flights[0][a2] * seats_type2 + AC3_flights[0][
                        a2] * seats_type3)
                    pax1 = xij[0][a2] + xij[a1][0] + wij[a1][a2]
                    cost1 = (cost_type1[0][a2] * AC1_flights[0][a2] + cost_type2[0][a2] * AC2_flights[0][a2] + \
                        cost_type3[0][a2] * AC3_flights[0][a2]) + (cost_type1[a1][0] * AC1_flights[a1][0] + \
                        cost_type2[a1][0] * AC2_flights[a1][0] + cost_type3[a1][0] * AC3_flights[a1][0])
                    revenue1 =  Yield[a1][a2] * dist1 * pax1
                    # revenue1 = (Yield[0][a2] * dist1 * xij[0][a2]) + (Yield[a1][0] * dist1 * xij[a1][0])
                    ASK1 = dist1 * seat1
                    RPK1 = dist1 * pax1 #(dist[a1][0] * xij[a1][0]) + (dist[0][a2] * xij[0][a2]) + 2 * (dist[a1][a2] * wij[a1][a2])
                    if ASK1 != 0:
                        RASK1 = revenue1 / ASK1
                        CASK1 = cost1 / ASK1
                    if RPK1 != 0:
                        Yield1 = revenue1 / RPK1
                    op1 = (RASK1 - CASK1) * ASK1
                    profittt.append(revenue1)
                    LF1 = RPK1 / ASK1
                    lf.append(LF1)
                    ask.append(ASK1)
                    rpk.append(RPK1)
                    rask.append(RASK1)
                    cask.append(CASK1)
                    yieldd.append(Yield1)
                    print('ASK:', ASK1)
                    print('RPK:', RPK1)
                    print('RASK:', RASK1)
                    print('CASK:', CASK1)
                    print('Yield:', Yield1)
                    print('Unit profit:', RASK1 - CASK1)
                    print('Operational profit: ', op1)
                    print('Load factor:', LF1)

print('ASK Total', np.sum(ask))
print('ASK mean', np.average(ask))
print('RPK Total', np.sum(rpk))
print('RPK mean', np.average(rpk))
print('RASK mean', totalrevenue/np.sum(ask))
print('CASK mean', totalcost/np.sum(ask))
print('LF mean', np.average(lf))
print('Profit', profit)
print('Unit profit mean', np.average(rask)-np.average(cask))
