from gurobipy import Model, GRB, quicksum
import pandas as pd

# Load data from 'Group_6.xlsx' into dataframes
file_path = '/Users/yehorfurtsev/Documents/TIL_First_Year/TIL_Second_Year/AE4423-20 Airline Planning and Optimisation/Assignment_1 /APO_assignments/Assignment_1_P2/Group_6.xlsx'  # Update with the correct path
flight_df = pd.read_excel(file_path, sheet_name='Flight')
itinerary_df = pd.read_excel(file_path, sheet_name='Itinerary')
recapture_rate_df = pd.read_excel(file_path, sheet_name='Recapture Rate')
aircraft_df = pd.read_excel(file_path, sheet_name='Aircraft')

# Define the sets and parameters
N = list(set(flight_df['ORG'].tolist() + flight_df['DEST'].tolist()))  # Set of nodes (airports)

# Flights originating and terminating at each node (airport)
O = {airport: flight_df[flight_df['ORG'] == airport]['Flight Number'].tolist() for airport in N}
I = {airport: flight_df[flight_df['DEST'] == airport]['Flight Number'].tolist() for airport in N}

# Initialize the model
m = Model('AirlineFleetOptimization')

# Decision Variables
f = m.addVars(flight_df['Flight Number'], aircraft_df['Type'], vtype=GRB.BINARY, name="f")
y = m.addVars(N, aircraft_df['Type'], vtype=GRB.INTEGER, lb=0, name="y")
t = m.addVars(itinerary_df['Itin No.'], itinerary_df['Itin No.'], vtype=GRB.INTEGER, lb=0, name="t")

# Parameters
AC_k = dict(zip(aircraft_df['Type'], aircraft_df['Units']))
SEATS_k = dict(zip(aircraft_df['Type'], aircraft_df['Seats']))
Q_i = dict(zip(itinerary_df['Itin No.'], itinerary_df['Demand']))

# Cost of assigning aircraft type k to flight i
c_ki = {(flight, ac_type): flight_df.loc[flight_df['Flight Number'] == flight, ac_type].values[0]
        for flight in flight_df['Flight Number']
        for ac_type in aircraft_df['Type']}

# Fare for each itinerary
fare_p = dict(zip(itinerary_df['Itin No.'], itinerary_df['Fare']))

# Recapture rate when reallocating passengers from itinerary p to r
b_rp = {(row['From Itinerary'], row['To Itinerary']): row['Recapture Rate']
        for _, row in recapture_rate_df.iterrows()}

# Objective Function
m.setObjective(
    quicksum(c_ki[flight_num, ac_type] * f[flight_num, ac_type]
             for flight_num in flight_df['Flight Number'] for ac_type in aircraft_df['Type']) +
    quicksum((fare_p[itin_p] - b_rp.get((itin_p, itin_r), 0) * fare_p[itin_r]) * t[itin_p, itin_r]
             for itin_p in itinerary_df['Itin No.'] for itin_r in itinerary_df['Itin No.']),
    GRB.MINIMIZE)

# Constraints
# Flight Schedule Coverage
for i in flight_df['Flight Number']:
    m.addConstr(quicksum(f[i, k] for k in aircraft_df['Type']) == 1, name=f"coverage_{i}")

# Balance at Each Node
for k in aircraft_df['Type']:
    for n in N:
        m.addConstr(quicksum(f[i, k] for i in O[n]) - quicksum(f[i, k] for i in I[n]) == y[n, k], name=f"balance_{n}_{k}")

# Number of Aircraft Available
for k in aircraft_df['Type']:
    m.addConstr(quicksum(y[n, k] for n in N) <= AC_k[k], name=f"aircraft_available_{k}")

# Seat Capacity and Demand
for i in flight_df['Flight Number']:
    m.addConstr(quicksum(SEATS_k[k] * f[i, k] for k in aircraft_df['Type']) >= Q_i[i], name=f"demand_{i}")

# Run the model
m.optimize()

# Print the solution
if m.status == GRB.OPTIMAL:
    for var in m.getVars():
        if var.x > 0:
            print(f'{var.varName}: {var.x}')
else:
    print("No optimal solution found.")
