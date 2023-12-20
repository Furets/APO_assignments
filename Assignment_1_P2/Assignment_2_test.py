#%% Import libraries
from gurobipy import *
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

#%% DATA IMPORT AND PARAMETER + SET DEFINITION

# Import data
sheet_names = ['Flight', 'Itinerary', 'Recapture Rate', 'Aircraft']
df = pd.read_excel('Group_5.xlsx', sheet_name=sheet_names)
flight_data, itin_data, recapture_data, aircraft_data = [df[name] for name in sheet_names]

# ------------------- Sets --------------------
T = np.linspace(0,1435,288)         # set of time periods
K = range(len(aircraft_data))       # set of aircrafts
L = range(len(flight_data))         # set of flights
P = range(len(itin_data)+1)         # set of all itineraries + fictuous

# Make list of all airports
unique_airports = pd.unique(flight_data[['ORG', 'DEST']].values.ravel('K'))
airport_list = unique_airports.tolist()

# Convert TAT to hours:minutes (same notation arrival and departure times)
TAT_k = list(aircraft_data['TAT'])
TAT_conv = [time(minute // 60, minute % 60) for minute in TAT_k]

# Add TAT to arrival times for all flights and aircraft types
A_i = [timedelta(hours=flight_data.iloc[i]['Arrival'].hour, minutes=flight_data.iloc[i]['Arrival'].minute) for i in L]
TAT_conv = [timedelta(hours=TAT_conv[k].hour, minutes=TAT_conv[k].minute) for k in K]
arrivals = [[(A_i[i] + TAT_conv[k]) for i in L] for k in K]
    
# Arrival and departure times for every flight, for every airport and every aircraft type, sorted
A_kni = []
for k in K:
    A_k = []
    for n in airport_list:
        A_i = set()
        for j in range(len(flight_data['ORG'])):
            if n == flight_data.iloc[j]['ORG']:
                departure_time = flight_data.iloc[j]['Departure']
                arrival_time = flight_data.iloc[j]['Arrival']
                if arrival_time > departure_time:
                    departure_delta = timedelta(hours=departure_time.hour, minutes=departure_time.minute, seconds=departure_time.second)
                    A_i.add(departure_delta)
            if n == flight_data.iloc[j]['DEST']:
                A_i.add(arrivals[k][j])
        A_i = sorted(A_i)
        A_k.append(A_i)
    A_kni.append(A_k)

Node, G, Gk = [], [], []
count_node, count_G, count_Ground_arcs = 0, 0, 0

for k in K:
    Node_k, G_k, Gk_k = [], [], []
    for n in range(len(airport_list)):
        size = len(A_kni[k][n])

        # Make set of nodes for all airports and all arrival times 
        Node_k.append(sorted(range(count_node, count_node + size)))
        count_node += size

        # Make set of ground arcs for every aircraft type k and airport n
        G_i = sorted(range(count_G, count_G + size + 1))
        G_k.append(G_i)
        count_G += size + 1

        # Make set of ground arcs G^k
        Gk_k.extend(range(count_Ground_arcs, count_Ground_arcs + size + 1))
        count_Ground_arcs += size + 1

    Node.append(Node_k)
    G.append(G_k)
    Gk.append(Gk_k)

# make set of all nodes combined with departure time and airport location
arcs = [[value for sublist in Node[k] for value in sublist] for k in K]
times = [[value for sublist in A_kni[k] for value in sublist] for k in K]
airports = [[airport_list[lists] for lists in range(len(Node[k])) for _ in range(len(Node[k][lists]))] for k in K]

combined_nodes = [[[arcs[k][j], times[k][j], airports[k][j]] for j in range(len(times[k]))] for k in K]

count_new = 0
n_plus = []
for k in K:
    n_plus_k = []
    for j, timeline in enumerate(G[k]):
        for i, value in enumerate(timeline):
            if i == 0:
                count_new += 1
                continue
            n_plus_value = count_new
            count_new += 1
            n_plus_k.append(n_plus_value)
    n_plus.append(n_plus_k)


# ground arcs terminating at any node n 
count_new = 0
n_minus = []
for k in K:
    n_minus_k = []
    for timeline in G[k]:
        for i, value in enumerate(timeline):
            n_minus_value = count_new
            if i == len(timeline) -1: # check if it is the last element of the timeline
                count_new += 1
                continue
            n_minus_value = count_new
            count_new += 1
            n_minus_k.append(n_minus_value)
    n_minus.append(n_minus_k)
    
# sets of originating and terminating flights at the different nodes
O = []
for k in K:
    O_k = []
    flight_data['Departure'] = pd.to_timedelta(flight_data['Departure'].astype(str))
    for n in range(len(times[k])):
        indices = flight_data.loc[(flight_data['ORG'] == combined_nodes[k][n][2].strip("'")) & (flight_data['Departure'] == combined_nodes[k][n][1])].index.tolist()
        O_k.append(indices)
    O.append(O_k)

I = []
for k in K:
    I_k = []
    flight_data['Arrival'] = pd.to_timedelta(flight_data['Arrival'].astype(str))
    for n in range(len(times[k])):
        indices = flight_data.loc[(flight_data['DEST'] == combined_nodes[k][n][2].strip("'")) & (flight_data['Arrival'] == (combined_nodes[k][n][1] - TAT_conv[k]))].index.tolist()
        I_k.append(indices)
    I.append(I_k)
    
# making set of flight arcs for time steps
flight_data['Arrival_minutes'] = flight_data['Arrival'].div(pd.Timedelta(seconds=60))
flight_data['Departure_minutes'] = flight_data['Departure'].div(pd.Timedelta(seconds=60))
for i, row in flight_data.iterrows():
    if row['Departure_minutes'] > row['Arrival_minutes']:
        flight_data.at[i,'Departure_minutes'] = -10

# set of feasible nodes at time steps
NF = [[[i for i, row in flight_data.iterrows() if row['Arrival_minutes'] + TAT_k[k] > T[t] and row['Departure_minutes'] < T[t]] for t in range(len(T))] for k in K]

# set of feasible ground nodes at time step
NG = []
for k in K:
    NG_k = []
    for t in range(len(T)):
        NG_t = []
        for index, i in enumerate(G[k]):
            for j, value in enumerate(i):
                if j == 0:
                    value_minutes = -10
                    value_next_minutes = A_kni[k][index][j].total_seconds()/60
                elif j == len(i) - 1:
                    value_minutes = A_kni[k][index][j-1].total_seconds()/60
                    value_next_minutes = 1450
                else:
                    value_minutes = A_kni[k][index][j-1].total_seconds()/60
                    value_next_minutes = A_kni[k][index][j].total_seconds()/60
                if value_minutes < T[t] and value_next_minutes > T[t]:
                    NG_t.append(value)
        NG_k.append(NG_t)
    NG.append(NG_k)

# set of all airports for any aircraft type k
N = []
for k in K:
    N_k = range(len(n_plus[k]))
    N.append(N_k)


# ------------------- Parameters --------------------
# c_ki calculation: costs of flying flight leg i with aircraft type k
c_ik = np.zeros((len(L), len(K)))
flight_numbers = flight_data['Flight Number'].tolist()
aircraft_types = ['A330', 'A340', 'B737', 'B738']
for index, row in flight_data.iterrows():
    flight_index = flight_numbers.index(row['Flight Number'])
    for i, aircraft in enumerate(aircraft_types):
        c_ik[flight_index, i] = row[aircraft]


AC_k = aircraft_data['Units']               # number of aircrafts available each aircraft type k
fare_p = itin_data['Fare']                  # average fare itinerary p
fare_p = pd.concat([pd.Series([0]), fare_p]).reset_index(drop=True)     # fare fictuous itinerary = 0
fare_r = fare_p                             # average fare itinerary r (same as p)
s = aircraft_data["Seats"]                  # number of seats on aircraft type k 
D = itin_data["Demand"]                     # daily unconstrained demand for itinerary p                    
D = pd.concat([pd.Series([0]), D]).reset_index(drop=True)   # demand fictuous itinerary = 0

# calculation b_pr: recapture rate from itinerary p to r
b_pr = np.zeros((len(P), len(P)))
for index, row in recapture_data.iterrows():
    from_itinerary = int(row['From Itinerary'])
    to_itinerary = int(row['To Itinerary'])
    recapture_rate = row['Recapture Rate']
    b_pr[from_itinerary, to_itinerary] = recapture_rate
np.fill_diagonal(b_pr, 1)
for p in P:
    b_pr[p,0] = 1
    
# calculation Q_i : daily unconstrained demand on flight leg i 
Q_i = np.zeros(len(L))
for i in itin_data.index: 
    for j in flight_data.index: 
        if itin_data["Leg 1"][i] == flight_data["Flight Number"][j]:
            Q_i[j] = Q_i[j] + itin_data["Demand"][i]
        if itin_data["Leg 2"][i] == flight_data["Flight Number"][j]:
            Q_i[j] = Q_i[j]+ itin_data["Demand"][i]

# calculation delta_ip: if flight i belongs to itinerary p 
delta_ip = np.zeros((len(L), len(P)))
for i in itin_data.index: 
    for j in flight_data.index: 
        if itin_data["Leg 1"][i] == flight_data["Flight Number"][j]:
            delta_ip[j][i] = 1
        if itin_data["Leg 2"][i] == flight_data["Flight Number"][j]:
            delta_ip[j][i] = 1
delta_ip = np.insert(delta_ip, 0, 0, axis=1)

# make set of flights that are in itinerary
flight_in_it = [[j for j, row in flight_data.iterrows() if row['Flight Number'] in [itin_data.loc[i, 'Leg 1'], itin_data.loc[i, 'Leg 2']]] for i in range(len(itin_data))]
flight_in_it.insert(0, [])


#%% INITIALIZATION + RMP

# Initialization: all spilled passengers are recaptured to fictuous itinerary 0
R = [0]                         # start with set R only fictuous itinerary
    
def RMP(R):
    model = Model('RMP')

    # --- Decision variables ---
    # if flight arc i is assigned to aircraft type k (1) or not (0)
    f = {(i, k): model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f[{i},{k}]') for i in L for k in K}
        
    # number of aircraft of type k on ground arc a 
    y = {(a, k): model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'y[{a},{k}]') for k in K for a in Gk[k]}

    # number of passengers that would like to travel on itinerary p and are reallocated by the airline to itinerary r            
    t = {(p, r): model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f't[{p},{r}]') for p in P for r in R}

    model.update()

    # --- Objective function ---
    model.setObjective (quicksum(c_ik[i,k] * f[i,k] for i in L for k in K) + quicksum(((fare_p[p]-b_pr[p,r]*fare_r[r])*t[p,r]) for p in P for r in R))
    model.modelSense = GRB.MINIMIZE
    model.update()

    # ---- Constraints ----
    # Every flight is covered exactly once by one aircraft of type k
    con1 = {i: model.addConstr(quicksum(f[i, k] for k in K) == 1, f'con1[{i}]') for i in L}

    # Node balance: Number of aircraft arriving is equal to departing number for every aircraft type k
    con2 = {(k, n): model.addConstr(y[n_plus[k][n], k] + quicksum(f[i, k] for i in O[k][n]) - y[n_minus[k][n], k] - quicksum(f[i, k] for i in I[k][n]) == 0, f'con2[{k},{n}]') for k in K for n in N[k]}

    # Limits on aircraft availability per time step
    con3 = {(k, t): model.addConstr(quicksum(y[a, k] for a in NG[k][t]) + quicksum(f[a, k] for a in NF[k][t]) <= AC_k[k], f'con3[{k},{t}]') for k in K for t in range(len(T))}

    # Capacity constraint: Capacity of flight i + pax removed + pax recaptured >= unconstrained demand flight leg i
    con4 = {i: model.addConstr(quicksum(s[k] * f[i, k] for k in K) + quicksum(delta_ip[i, p] * t[p, r] for p in P for r in R) - quicksum(delta_ip[i, p] * b_pr[r, p] * t[r, p] for r in P for p in R) >= Q_i[i], f'con4[{i}]') for i in L}

    # Recapture constraint: Number of passengers spilled or recaptured should be lower than demand itinerary
    con5 = {p: model.addConstr(quicksum(t[p, r] for r in R) <= D[p], f'con5[{p}]') for p in P}

    model.update()

    #---- Solve ----
    model.setParam( 'OutputFlag', True)
    model.setParam ('MIPGap', 0.0001);
    model.setParam('TimeLimit', 100)
    model.write("output.lp")

    model.optimize()
    print('The runtime is: ', model.Runtime)

    return model

# var_list = [(variable.varName, variable.x) for variable in model.getVars()]
#%% COLUMN GENERATION ALGORITHM

def columngeneration(max_iterations, R):
    iteration = 0
    
    while iteration < max_iterations:    # stop algorithm if maximum nr of iterations reached
        
        model = RMP(R)                          # solve RMP with current columns
        pi = [c.Pi for c in model.getConstrs()] # get dual variables for constraints
    
        pi_i = pi[2802:3010]                    # dual variables corresponding to constraint 4
        sigma_p = pi[3010:3565]                 # dual variables corresponding to constraint 5
        
        # calculatate c_pr
        c_pr = np.zeros((len(P), len(P)))
        for p in P:
            for r in P:
                c_pr_value = (fare_p[p] - sum(pi_i[i] for i in flight_in_it[p])) - b_pr[p][r] * (fare_r[r] - sum(pi_i[j] for j in flight_in_it[r])) - sigma_p[p]
                c_pr[p][r] = c_pr_value
    
        # add columns with negative c_pr
        addition_counter = 0
        for p in P:
            for r in P:
                if c_pr[p][r] < 0 and b_pr[p][r] != 0 and r not in R:
                    print(addition_counter)
                    R.append(r)
                    addition_counter += 1
                  
        # stop algorithm if there is no decision variables to be added
        if addition_counter == 0:
            break
        
        print('Iteration ', iteration)
        print(R)
        iteration += 1      # go on to next iteration
        
    return R

#%% FINAL SOLUTION - NON-RELAXED MODEL WITH COLUMN GENERATION DECISION VARIABLES

model = Model('Final solution')
R = columngeneration(max_iterations=20, R = [0])

# --- Decision variables ---
# if flight arc i is assigned to aircraft type k (1) or not (0)
f = {(i, k): model.addVar(lb=0, vtype=GRB.BINARY, name=f'f[{i},{k}]') for i in L for k in K}
        
# number of aircraft of type k on ground arc a 
y = {(a, k): model.addVar(lb=0, vtype=GRB.INTEGER, name=f'y[{a},{k}]') for k in K for a in Gk[k]}

# number of passengers that would like to travel on itinerary p and are reallocated by the airline to itinerary r            
t = {(p, r): model.addVar(lb=0, vtype=GRB.INTEGER, name=f't[{p},{r}]') for p in P for r in R}

model.update()

# --- Objective function ---
model.setObjective (quicksum(c_ik[i,k] * f[i,k] for i in L for k in K) + quicksum(((fare_p[p]-b_pr[p,r]*fare_r[r])*t[p,r]) for p in P for r in R))
model.modelSense = GRB.MINIMIZE
model.update()

# ---- Constraints ----
# Every flight is covered exactly once by one aircraft of type k
con1 = {i: model.addConstr(quicksum(f[i, k] for k in K) == 1, f'con1[{i}]') for i in L}

# Node balance: Number of aircraft arriving is equal to departing number for every aircraft type k
con2 = {(k, n): model.addConstr(y[n_plus[k][n], k] + quicksum(f[i, k] for i in O[k][n]) - y[n_minus[k][n], k] - quicksum(f[i, k] for i in I[k][n]) == 0, f'con2[{k},{n}]') for k in K for n in N[k]}

# Limits on aircraft availability per time step
con3 = {(k, t): model.addConstr(quicksum(y[a, k] for a in NG[k][t]) + quicksum(f[a, k] for a in NF[k][t]) <= AC_k[k], f'con3[{k},{t}]') for k in K for t in range(len(T))}

# Capacity constraint: Capacity of flight i + pax removed + pax recaptured >= unconstrained demand flight leg i
con4 = {i: model.addConstr(quicksum(s[k] * f[i, k] for k in K) + quicksum(delta_ip[i, p] * t[p, r] for p in P for r in R) - quicksum(delta_ip[i, p] * b_pr[r, p] * t[r, p] for r in P for p in R) >= Q_i[i], f'con4[{i}]') for i in L}

# Recapture constraint: Number of passengers spilled or recaptured should be lower than demand itinerary
con5 = {p: model.addConstr(quicksum(t[p, r] for r in R) <= D[p], f'con5[{p}]') for p in P}

model.update()

#---- Solve ----
model.setParam( 'OutputFlag', True)
model.setParam ('MIPGap', 0.0001);
model.setParam('TimeLimit', 100)
model.write("output.lp")

model.optimize()
print('The runtime is: ', model.Runtime)

# --- Print results ---
print ('\n--------------------------------------------------------------------\n')

#%%
if True:
    #print ('Minimum lost cost : %10.2f euro' % model.objVal)
    print ('')
    print ('All decision variables:\n')
    
    print ( 'f_{ik}' )
    o = '%10s' % ''
    for k in K:
        o = o + '%12s' % k
    print (o)    
    
    for i in L:
        o = '%10s' % i
        for k in K:
            o = o + '%12f' % f[i,k].x
        o = o + '%15f' % sum (f[i,k].x for k in K)    
        print (o)    
    
    o = '%10s' % ''
    for k in K:
        o = o + '%12f' % sum (f[i,k].x for i in L)    
    print (o)
    
    print ( 'y_{ak}' )
    o = '%10s' % ''
    for k in range(445):
        o = o + '%12s' % k
    print (o)    

    for k in K:
        o = '%10s' % k    
        for i in Gk[k]:
            o = o + '%12f' % y[i,k].x
        o = o + '%15f' % sum (y[i,k].x for i in Gk[k])    
        print (o)    
    
    # o = '%10s' % ''
    # for i in Gk[k]:
    #     o = o + '%12f' % sum (y[i,k].x for k in K)    
    # print (o)
    
    print ( 't_{pr}' )
    
    o = '%10s' % ''
    for r in R:
        o = o + '%12s' % r
    print (o)  
    
    for p in range(50):
        o = '%10s' % p
        for r in R:
            o = o + '%12f' % t[p,r].x  
        o = o + '%15f' % sum (t[p,r].x for r in R)
        print (o)    
    
    o = '%10s' % ''
    for r in R:
        o = o + '%12f' % sum (t[p,r].x for p in P)    
    print (o)

var_list2 = [(variable.varName, variable.x) for variable in model.getVars()]

#%%
t_results = []
for p in P:
    for r in R:
        if t[p,r].x != 0:
            t_results_i = []
            t_results_i.append('t[' + str(p) +  ',' + str(r) + ']')
            t_results_i.append(t[p,r].x)
            t_results.append(t_results_i)

for i in range(len(t_results)):
               print(t_results[i])