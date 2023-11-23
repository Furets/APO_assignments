## Preprocessing file
import numpy as np
import scipy.optimize as optimize
import pandas as pd
from math import *
import xlsxwriter

# Read data from csv file
data = pd.read_csv('Assignment1_data.csv', skiprows=4)

header_row = 1
data.columns = data.iloc[header_row]
data = data.drop(header_row)
data = data.reset_index(drop=True)

demanddata = pd.read_csv('Assignment1_data.csv', skiprows=11)
demanddata.columns = demanddata.iloc[header_row]
demanddata = demanddata.drop(header_row)
demanddata = demanddata.reset_index(drop=True)

# Create results file
workbook = xlsxwriter.Workbook('results.xlsx')

# Fuel cost [USD/gallon]
fuel_price = 1.42

airports = ['ESGG', 'ESPA', 'ESMS', 'ESSA', 'ESTA', 'ESNZ', 'ESNX', 'ESGJ', 'ESQO', 'ESKV']  # 2020
futureairports = ['ESGG', 'ESMS', 'ESPA', 'ESSA', 'ESFR', 'ESDF', 'ESIB', 'SE-0016', 'ESNS', 'ESGR', 'ESNY', 'ESKN',
                  'ESNB', 'ESCM', 'ESGO']  # 2030

# Creating a matrix (15x15) for demand
demand = [[0 for i in range(15)] for j in range(15)]
# Creating a matrix (15x15) for distances
distances = [[0 for i in range(15)] for j in range(15)]

# Weekly demand in 2020
def realDemand(ICAO_orig, ICAO_dest):
    rownumber = airports.index(ICAO_orig) + 1
    columnnumber = airports.index(ICAO_dest) + 2
    # Extract de demand from the table from csv file (Assignment1_data.csv)
    stat_demand = float(demanddata.iat[rownumber, columnnumber])

    return stat_demand

# Determine the distance between airport i and airport j
def distance(ICAO_orig, ICAO_dest):
    orig_lat = (float(data.loc[1, [str(ICAO_orig)]].astype(float)) * np.pi) / 180
    orig_lon = (float(data.loc[2, [str(ICAO_orig)]].astype(float)) * np.pi) / 180
    dest_lat = (float(data.loc[1, [str(ICAO_dest)]].astype(float)) * np.pi) / 180
    dest_lon = (float(data.loc[2, [str(ICAO_dest)]].astype(float)) * np.pi) / 180
    arclength = 2 * np.arcsin(np.sqrt((np.sin((orig_lat - dest_lat) / 2)) ** 2 + np.cos(orig_lat)
                                      * np.cos(dest_lat) * (np.sin((orig_lon - dest_lon) / 2)) ** 2))
    # Radius of the Earth
    R_e = 6371.  # [km]
    distance = R_e * arclength

    return distance

# Determine the demand in 2020 based on gravity model
def predictLogDemand(ICAO_orig, ICAO_dest, b1, b2, k):
    # Population of original airport
    pop_orig = float(data.loc[4, [str(ICAO_orig)]].astype(float))
    # Population of destination airport
    pop_dest = float(data.loc[4, [str(ICAO_dest)]].astype(float))

    # Distance between original airport and destination airport
    dist = distance(ICAO_orig, ICAO_dest)

    # Applying logarithms
    d_ij = np.log(k) + b1 * np.log(pop_orig * pop_dest) - b2 * np.log(fuel_price * dist)

    return d_ij

def forecastLogDemand(ICAO_orig, ICAO_dest, b1, b2, k):
    # Annual population growth
    ap = 0.8/100
    # Population of original airport
    pop_orig = float(data.loc[4, [str(ICAO_orig)]].astype(float)) * ((1 + ap) ** 10)
    # Population of destination airport
    pop_dest = float(data.loc[4, [str(ICAO_dest)]].astype(float)) * ((1 + ap) ** 10)

    dist = distance(ICAO_orig, ICAO_dest)

    # Applying logarithms
    d_ij = np.log(k) + b1 * np.log(pop_orig * pop_dest) - b2 * np.log(fuel_price * dist)

    return d_ij

# Determine de RMSE (Root-mean-square deviation) and cost
def costFunction(x):
    b1, b2, k = x
    totalCost = 0

    for i in airports:
        for j in airports:
            if i != j:
                statdemand = np.log(realDemand(i, j))
                actualdem = predictLogDemand(i, j, b1, b2, k)
                totalCost += (statdemand - actualdem) ** 2

    rmse = np.sqrt(totalCost / len(airports))
    return rmse

x0 = (0.35, 0.25, 0.3)  # initial condition
bounds = ((-5, 5), (-5, 5), (0.0001, 10))
res = optimize.minimize(costFunction, x0, method="L-BFGS-B", bounds=bounds)
b1, b2, k = res.x

# Create a sheet which contains the scaling factor k and the parameters b1 b2
worksheet = workbook.add_worksheet('Scaling factor & parameters')
worksheet.write('A1', 'k')
worksheet.write('A2', k)
worksheet.write('B1', 'b1')
worksheet.write('B2', b1)
worksheet.write('C1', 'b2')
worksheet.write('C2', b2)


# Create a sheet which contains the distance between airport i and airport j
# Distances matrix with dimension: 15x15
for i in range(len(futureairports)):
    for j in range(len(futureairports)):
        airportdistance = distance(futureairports[i], futureairports[j])
        distances[i][j] = airportdistance

worksheet = workbook.add_worksheet('Distances')
i = 1
j = 1
for item in futureairports:
    worksheet.write(i, 0, item)
    worksheet.write(0, j, item)
    i += 1
    j += 1

for i in range(1, len(futureairports)+1):
    for j in range(1, len(futureairports)+1):
        worksheet.write(i, j, distances[i-1][j-1])

# Create a sheet which contains the forecast demand between airport i and aiport j
# Demand matrix with dimension: 15x15
for i in range( len(futureairports)):
    for j in range(len(futureairports)):
        if i != j:
            futuredemand = ceil(np.exp(forecastLogDemand(futureairports[i], futureairports[j], b1, b2, k)))
            demand[i][j] = futuredemand

worksheet = workbook.add_worksheet('Forecast demand')
i = 1
j = 1
for item in futureairports:
    worksheet.write(i, 0, item)
    worksheet.write(0, j, item)
    i += 1
    j += 1

for i in range(1, len(futureairports)+1):
    for j in range(1, len(futureairports)+1):
        worksheet.write(i, j, demand[i-1][j-1])

# Close the excel file
workbook.close()