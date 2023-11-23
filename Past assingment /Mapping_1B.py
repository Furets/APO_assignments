#######################################################################################################################
##############################################   INSTRUCTIONS   #######################################################
#######################################################################################################################

# TO RUN THIS FILE, RUN EACH SECTION SEPARATELY (SEE COMMENTS)
# e.g. if mapping of full network is desired, comment sections for aircraft 1, 2, 3
# TO VISUALISE RESULTS (i.e. TO SEE THE MAP WITH THE NETWORK), OPEN 'indexy.html' FILE IN BROWSER OF PREFERENCE

#######################################################################################################################

import numpy as np
import pandas as pd
import folium

# Getting airport locations data from CSV file 'Assignment1_data_for_mapping.csv'
data = pd.read_csv("Assignment1_data_for_mapping.csv",skiprows=6,nrows=2)
data = data.set_index('ICAO_Code').drop(data.columns[-7:], axis=1).drop(data.columns[0], axis=1)
dictionary = data.to_dict()

# Creating lists with longitudes and latitudes of each airport
longs = []
lats = []
for key in list(dictionary.keys()):
    longs.append(dictionary[key]['Longitude_(deg)'])
    lats.append(dictionary[key]['Latitude_(deg)'])

# Creating pairs of longitudes and latitudes (i.e. coordinates) for each airport
tuples = []
for i in range(len(lats)):
    tuples.append((lats[i],longs[i]))

# Plotting airports on map
m = folium.Map(location=[61,15],zoom_start=6)
for each in tuples:
    folium.Marker(each,color = "red").add_to(m)

# Extracting data from optimisation results (Excel file 'Model_1B_results.xls')
xls = pd.ExcelFile('Model_1B_results.xls')
x_ij = pd.read_excel(xls, 'x') # direct flows
w_ij = pd.read_excel(xls, 'w') # flows through hub
ac_type_1 = pd.read_excel(xls, 'AC 1') # frequency of aircraft type 1
ac_type_2 = pd.read_excel(xls, 'AC 2') # frequency of aircraft type 2
ac_type_3 = pd.read_excel(xls, 'AC 3') # frequency of aircraft type 3

# Creating matrices with extracted data for each decision variable
direct_network = np.array(x_ij)[::,1::]
transfer_network = np.array(w_ij)[::,1::]
AC_1_network = np.array(ac_type_1)[::,1::]
AC_2_network = np.array(ac_type_2)[::,1::]
AC_3_network = np.array(ac_type_3)[::,1::]

# Airports ICAO codes
Airports = ['ESGG', 'ESMS', 'ESPA', 'ESSA', 'ESFR', 'ESDF', 'ESIB', 'SE-0016', 'ESNS', 'ESGR', 'ESNY', 'ESKN', 'ESNB', 'ESCM', 'ESGO']

#######################################################################################################################
##############################################   RUN FOR FULL NETWORK   ###############################################
#######################################################################################################################

# DIRECT FLOWS
# Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
dicti_x_ij = {}
for i in range(len(direct_network)):
    for j in range(len(direct_network)):
        x_routes = direct_network[i][j]
        if x_routes != 0:
            if Airports[i] not in dicti_x_ij:
                dicti_x_ij[Airports[i]] = []
            dicti_x_ij[Airports[i]].append(Airports[j])

# Plotting on map
for airport in Airports:
    if airport in dicti_x_ij:
        for i in dicti_x_ij[airport]:
            lista = [(dictionary[airport]['Latitude_(deg)'],dictionary[airport]['Longitude_(deg)']),(dictionary[i]['Latitude_(deg)'],dictionary[i]['Longitude_(deg)'])]
            folium.PolyLine(lista, color="red", weight=2.5, opacity=1).add_to(m)

# FLOWS THROUGH HUB
# Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
dicti_w_ij = {}
for i in range(len(transfer_network)):
    for j in range(len(transfer_network)):
        w_routes = transfer_network[i][j]
        if w_routes != 0:
            if Airports[i] not in dicti_w_ij:
                dicti_w_ij[Airports[i]] = []
            dicti_w_ij[Airports[i]].append(Airports[j])

# Plotting on map
for airport in Airports:
    if airport in dicti_w_ij:
        for i in dicti_w_ij[airport]:
            lista_1 = [(dictionary[airport]['Latitude_(deg)'],dictionary[airport]['Longitude_(deg)']),(dictionary['ESGG']['Latitude_(deg)'],dictionary['ESGG']['Longitude_(deg)'])]
            lista_2 = [(dictionary['ESGG']['Latitude_(deg)'],dictionary['ESGG']['Longitude_(deg)']),(dictionary[i]['Latitude_(deg)'],dictionary[i]['Longitude_(deg)'])]
            folium.PolyLine(lista_1, color="red", weight=2.5, opacity=1).add_to(m)
            folium.PolyLine(lista_2, color="red", weight=2.5, opacity=1).add_to(m)
m.save('indexy.html')

#######################################################################################################################
##############################################   RUN FOR AIRCRAFT 1 NETWORK   #########################################
#######################################################################################################################

# # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
# dicti_AC_1 = {}
# for i in range(len(AC_1_network)):
#     for j in range(len(AC_1_network)):
#         AC1_routes = AC_1_network[i][j]
#         if AC1_routes != 0:
#             if Airports[i] not in dicti_AC_1:
#                 dicti_AC_1[Airports[i]] = []
#             dicti_AC_1[Airports[i]].append(Airports[j])
#
# # Plotting on map
# for airport in Airports:
#     if airport in dicti_AC_1:
#         for i in dicti_AC_1[airport]:
#             lista = [(dictionary[airport]['Latitude_(deg)'],dictionary[airport]['Longitude_(deg)']),(dictionary[i]['Latitude_(deg)'],dictionary[i]['Longitude_(deg)'])]
#             folium.PolyLine(lista, color="red", weight=2.5, opacity=1).add_to(m)
# m.save('indexy.html')
#
#######################################################################################################################
##############################################   RUN FOR AIRCRAFT 2 NETWORK   #########################################
#######################################################################################################################
#
# # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
# dicti_AC_2 = {}
# for i in range(len(AC_2_network)):
#     for j in range(len(AC_2_network)):
#         AC2_routes = AC_2_network[i][j]
#         if AC2_routes != 0:
#             if Airports[i] not in dicti_AC_2:
#                 dicti_AC_2[Airports[i]] = []
#             dicti_AC_2[Airports[i]].append(Airports[j])
#
# # Plotting on map
# for airport in Airports:
#     if airport in dicti_AC_2:
#         for i in dicti_AC_2[airport]:
#             lista = [(dictionary[airport]['Latitude_(deg)'],dictionary[airport]['Longitude_(deg)']),(dictionary[i]['Latitude_(deg)'],dictionary[i]['Longitude_(deg)'])]
#             folium.PolyLine(lista, color="red", weight=2.5, opacity=1).add_to(m)
# m.save('indexy.html')
#
#######################################################################################################################
##############################################   RUN FOR AIRCRAFT 3 NETWORK   #########################################
#######################################################################################################################
#
# # Creating dictionary associating each airport (origin) to a list of ALL its feasible destinations
# dicti_AC_3 = {}
# for i in range(len(AC_3_network)):
#     for j in range(len(AC_3_network)):
#         AC1_routes = AC_3_network[i][j]
#         if AC1_routes != 0:
#             if Airports[i] not in dicti_AC_3:
#                 dicti_AC_3[Airports[i]] = []
#             dicti_AC_3[Airports[i]].append(Airports[j])
#
# # Plotting on map
# for airport in Airports:
#     if airport in dicti_AC_3:
#         for i in dicti_AC_3[airport]:
#             lista = [(dictionary[airport]['Latitude_(deg)'],dictionary[airport]['Longitude_(deg)']),(dictionary[i]['Latitude_(deg)'],dictionary[i]['Longitude_(deg)'])]
#             folium.PolyLine(lista, color="red", weight=2.5, opacity=1).add_to(m)
# m.save('indexy.html')