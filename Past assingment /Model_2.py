from preprocessing import *
import numpy as np
from gurobipy import *
import timeit
from math import *
from xlwt import Workbook

start = timeit.default_timer()

#b1, b2, K = leastsquaresparameters()
b1 = 0.47324986070369485  # 0.31584784131337623
b2 =0.16721962529823242 # .26352185387317956
K = 0.004043484195831681  #0.2764624818703978
LF = 0.8
BT = 10 * 60 * 7 

print('b1: ', b1, 'b2: ',b2, 'k: ',K)
 
airports = ['ESGG','ESMS','ESPA','ESSA','ESFR','ESDF','ESIB','SE-0016','ESNS','ESGR','ESNY','ESKN','ESNB','ESCM','ESGO']
runway_lengths = [3299, 2800, 3350, 3301, 1987, 2331, 2264, 2500, 2520, 1736, 2524, 2878, 820, 1963, 890]

nodes = ['ESMS','ESPA','ESSA','ESFR','ESDF','ESIB','SE-0016','ESNS','ESGR','ESNY','ESKN','ESNB','ESCM','ESGO']
#   0 speed, 1 seats, 2 TAT, 3 charging time, 4 max range, 5 runway, 6 wkly lease, 7 fixed operating cost, 8 time cost, 9 fuel cost, 10 batt energy
k_aircraft = [[550, 45, 25, 0, 1500, 1400, 15000, 300, 750, 1, 0],
              [820,70,35,0,3300,1600,34000,600,775,2,0],
              [850,150,45,0,6300,1800,80000,1250,1400,3.75,0],
              [350,20,20,20,400,750,12000,90,750,0,2130],
              [480,48,25,45,1000,950,22000,120,750,0,8216]]

distances = np.zeros((len(airports),len(airports)))

for i in range(len(airports)):
    for j in range(len(airports)):
        airportdistance = distance(airports[i],airports[j]) 
        distances[i,j] = airportdistance
                       
routes = []

for i in range(len(airports)):
    if i != 0:
        routes.append([airports[0], airports[i], airports [0]])
        for j in range(len(airports)):
            if i != j:
                if j != 0:
                    routes.append([airports[0], airports[i], airports[j], airports [0]])


routelengths = [] 
for i in range(len(routes)):
    if len(routes[i]) == 3:
        routelength = distance(routes[i][0],routes[i][1]) + distance(routes[i][1],routes[i][2])
        routelengths.append(routelength)
    if len(routes[i]) == 4:
        routelength = distance(routes[i][0],routes[i][1]) + distance(routes[i][1],routes[i][2]) + distance(routes[i][2],routes[i][3])
        routelengths.append(routelength)

            
x = np.zeros((len(airports),len(airports)))
Yield = np.zeros((len(airports),len(airports)))
dij = np.zeros((len(airports),len(airports)))
q = np.zeros((len(airports),len(airports)))

a2 = {}
for r in range(len(routes)):
    for k in range(5):
        
        if k_aircraft[k][4] >= routelengths[r]:
            a2[r,k] = 10000
        else:
            a2[r,k] = 0

run = {}
for r in range(len(routes)):
    for k in range(5):

        runwaylengths = []
        
        for i in range(len(routes[r])):
            index = airports.index(routes[r][i])
            runwaylengths.append(runway_lengths[index])

        minrunwaylength  = min(runwaylengths)

        if minrunwaylength >= k_aircraft[k][5]:
            run[r,k] = 100000
        else:
            run[r,k] = 0


for i in range(len(airports)):
    for j in range(len(airports)):
        if i != j:
            dij[i][j] = distance(airports[i], airports[j])

for i in range(len(airports)):
    for j in range(len(airports)):
        if i != j:
            Yield[i][j] = (5.9*dij[i][j]**(-0.76)+0.043)


q = demand

delta = np.zeros((len(airports), len(airports), len(routes)))
                 

for r  in range(len(routes)):
    for x in range(len(routes[r])):
        for y in range(len(routes[r])):
            if y > x:
                if routes[r][x] != routes[r][y]:
                    i = airports.index(routes[r][x])
                    j = airports.index(routes[r][y])
                    delta[i][j][r] =1


S = [] #np.zeros((len(routes)))
P =  [] # np.zeros((len(routes)))

for r in range(len(routes)):
    templist = []
    templist2 = []
    for x in range(len(routes[r])-1):
        templist.append(routes[r][x+1:])
    for x in range(len(routes[r])-1):
        xlist = routes[r][:(x+1)]
        xlist = xlist[::-1]
        templist2.append(xlist)
    S.append(templist)
    P.append(templist2)


LTO = np.zeros((len(routes), 5))

for r in range(len(routes)):
    for k in range(5):
        if len(routes[r]) == 3:
            if k < 3:
                TAT = k_aircraft[k][2] * 2 + k_aircraft[k][2] * 1.5
            if k > 2:
                TAT = 3 * k_aircraft[k][2]
        if len(routes[r]) == 4:
            if k < 3:
                TAT = k_aircraft[k][2] * 3 + k_aircraft[k][2] * 1.5
            if k > 2:
                TAT = 4 * k_aircraft[k][2]

        LTO[r,k] = TAT


cost = np.zeros((len(routes),5))
for i in range(len(routes)):
    if len(routes[i]) == 3:
        routelength = distance(routes[i][0],routes[i][1]) + distance(routes[i][1],routes[i][2])
    if len(routes[i]) == 4:
        routelength = distance(routes[i][0],routes[i][1]) + distance(routes[i][1],routes[i][2]) + distance(routes[i][2],routes[i][3])

    for k in range(5):

        #if the route is a triangle, it has length 4
        #if so, the fixed costs and time based costs are partially discounted, partially not
        if len(routes[i]) == 4:
            fixed_costs = k_aircraft[k][7] * 2 *0.7 + 1 * k_aircraft[k][7]
            awaydistance = distance(routes[i][1],routes[i][2])
            timebased_costs = 0.7* k_aircraft[k][8] * (routelength-awaydistance) / k_aircraft[k][0]  +  k_aircraft[k][8] * awaydistance / k_aircraft[k][0] 

        #if length is 3, it's back and forth from the hub. Therefore, both costs are discounted.
        if len(routes[i]) == 3:
            fixed_costs = k_aircraft[k][7] * 2 *0.7
            timebased_costs = 0.7* k_aircraft[k][8] * routelength / k_aircraft[k][0] 

        
        timebased_costs = k_aircraft[k][8] * routelength / k_aircraft[k][0]

        if k < 3: 
            fuelcost1 = 0
            awaydistance = 0
            
            if len(routes[i]) == 4: # reduced costs are only 
                awaydistance = distance(routes[i][1],routes[i][2])
                fuelcost1 = k_aircraft[k][9] * 1.42 / 1.5 * routelength
                
            fuel_cost = 0.7 * k_aircraft[k][9] * 1.42 / 1.5 * (routelength-awaydistance) + fuelcost1

        if k > 2:
            fuel_cost = 0.07 * k_aircraft[k][10] * routelength / k_aircraft[k][4]


        cost[i][k] = fixed_costs+timebased_costs + fuel_cost


m = Model('Question 2')

x = {}
w = {}
z = {}


print("Now adding objective function:" ) 
for i in range(len(airports)):
    for j in range(len(airports)):

        for r in range(len(routes)):
            x[i,j,r] = m.addVar(obj = Yield[i][j] * dij[i][j], lb =0, vtype = GRB.INTEGER, name = 'x[%s,%s,%s]' %(i,j,r))
            
            for n in range(len(routes)):
                w[i,j,r,n] = m.addVar(obj = Yield[i][j] * dij[i][j] *  0.9, lb =0, vtype = GRB.INTEGER, name = 'w[%s,%s,%s,%s]' %(i,j,r,n))

for r in range(len(routes)):
    for k in range(5):
                z[r,k] = m.addVar(obj =  -cost[r][k], lb =0, vtype = GRB.INTEGER, name = 'z[%s,%s]' %(r,k))


AC = {}
for k in range(5):
    AC[k] = m.addVar(obj = -k_aircraft[k][6],  lb =0, vtype = GRB.INTEGER, name = 'AC[%s]' %(k))
    

m.update()
m.setObjective(m.getObjective(), GRB.MAXIMIZE)


print("Now adding constraints:")


for i in range(len(airports)):
    for j in range(len(airports)):
        m.addConstr(quicksum(x[i,j,r] + quicksum(w[i,j,r,n] for n in range(len(routes))) for r in range(len(routes))), GRB.LESS_EQUAL, q[i][j])

        for r in range(len(routes)):
            m.addConstr(x[i,j,r], GRB.LESS_EQUAL, delta[i][j][r] * q[i][j])

            for n in range(len(routes)):
                m.addConstr(w[i,j,r,n], GRB.LESS_EQUAL,  q[i][j] * delta[i][0][r] * delta[0][j][n])

    print("Progress on the demand constraint construction: ", (i +1)/ len(airports) * 100 , "%")

for r in range(len(routes)):
        #From the hub node constraint: 
        m.addConstr( quicksum(x[0,airports.index(S[r][0][m]),r] for m in range(len(routes[r])-1)) +
                     quicksum(quicksum(quicksum(w[p+1,airports.index(S[r][0][m]),r,n] for m in range(len(routes[r])-1)) for p in range(len(nodes)))   for n in range(len(routes))),
                     GRB.LESS_EQUAL, quicksum(z[r,k] * floor( k_aircraft[k][1] * LF ) for k in range(5)))

        m.addConstr( quicksum(x[airports.index(S[r][0][0]),airports.index(S[r][1][m]),r ] for m in range(len(S[r][1])))
                     + quicksum(x[airports.index(P[r][len(routes[r])-2][m]),airports.index(S[r][1][0]),r] for m in range(len(routes[r])-1))
                     + quicksum(quicksum(w[airports.index(S[r][1][0]),p+1,r,n] for p in range(len(nodes)))   for n in range(len(routes)))
                     + quicksum(quicksum(w[p+1,airports.index(S[r][0][0]),n,r] for p in range(len(nodes)))   for n in range(len(routes))), GRB.LESS_EQUAL, quicksum( z[r,k] * floor( k_aircraft[k][1] * LF ) for k in range(5))) 

        #To the hub node constraint:
        m.addConstr(quicksum(x[airports.index(P[r][len(routes[r])-2][m]),0,r]for m in range(len(routes[r])-1)) + quicksum(quicksum(quicksum(w[airports.index(P[r][len(routes[r])-2][m]),p+1,r,n] for m in range(len(routes[r])-1)) for p in range(len(nodes)))   for n in range(len(routes))), GRB.LESS_EQUAL, quicksum( z[r,k] * floor( k_aircraft[k][1] * LF ) for k in range(5)))

        for k in range(5):
            m.addConstr( z[r,k], GRB.LESS_EQUAL,  a2[r,k])

            m.addConstr( z[r,k], GRB.LESS_EQUAL, run[r,k])


for k in range(5):
    m.addConstr( quicksum( (routelengths[r] / (k_aircraft[k][0]/60) +  LTO[r,k] + k_aircraft[k][3]) * z[r,k]  for r in range(len(routes))) , GRB.LESS_EQUAL , BT * AC[k])


m.update()
m.setParam('TimeLimit', 60 * 60)

print("Optimizing: ")
m.optimize()

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

print("")
for k in range(5):
	print("Number of aircraft of type ",k, ":", AC[k].x)

print("")

for r in range(len(routes)):
    for k in range(5):
        if z[r,k].X >0:
            print("Route ", routes[r], "with amount of flights : ", z[r,k].X)

print("")

demandmatrix = np.zeros((len(airports), len(airports)))
for i in range(len(airports)):
	for j in range(len(airports)):
		directdemand = 0
		indirectdemand = 0
		for r in range(len(routes)):
			directdemand= directdemand + x[i,j,r].x
			for n in range(len(routes)):
				indirectdemand = indirectdemand + w[i,j,r,n].x
		demandmatrix[i,j] = demandmatrix[i,j] + directdemand + indirectdemand
		
		print("flight leg: ",airports[i], "to", airports[j] ,"Q: ",q[i][j], "fullfilled direct demand: ", directdemand, "transfers: ", indirectdemand )

print("")

capacitymatrix = np.zeros((len(airports), len(airports)))
for i in range(len(airports)):
	for j in range(len(airports)):
		for r in range(len(routes)):
			if airports[i] in routes[r]:
				if airports[j] in routes[r]:
					index_i = routes[r].index(airports[i])
					index_j = routes[r].index(airports[j])
					#print(index_i, index_j)
					if index_j == 0:
						index_j = len(routes[3])-1
					if index_j > index_i: 
						for k in range(5):
							if z[r,k].x > 0.001:
								capacitymatrix[i,j] = capacitymatrix[i,j] + z[r,k].x * k_aircraft[k][1]
								
	print("Capacity on flight leg ", airports[i], airports[j], " = ", capacitymatrix[i,j])


print("")


for i in range(len(airports)):
	for j in range(len(airports)):
		if demandmatrix[i,j] > 0 or capacitymatrix[i,j] > 0:
			print("flight leg : ", airports[i], " - ", airports[j], "\t" , "  with demand: ", demandmatrix[i,j], "and capacity ", capacitymatrix[i,j]) ## , "\t with load factor: ", demandmatrix[i,j]/capacitymatrix[i,j])

newx = np.zeros((len(airports), len(airports)))
neww = np.zeros((len(airports), len(airports)))


for i in range(len(airports)):
    for j in range(len(airports)):
        for r in range(len(routes)):
            newx[i,j] = newx[i,j] + x[i,j,r].x
            for n in range(len(routes)):
                neww[i,j] = neww[i,j] + w[i,j,r,n].x

num_ac_types = 5              
wb = Workbook()
overview = wb.add_sheet('Overview', cell_overwrite_ok = True)
sheetx = wb.add_sheet('x', cell_overwrite_ok = True)
sheetw = wb.add_sheet('w', cell_overwrite_ok = True)
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

# Writing results in the Excel file 'Model_2_results.xls'
sheetx.write(0, 0, 'x[i,j]')
sheetw.write(0, 0, 'w[i,j]')

for k in range(num_ac_types):
    locals()['sheet_{0}'.format(k)].write(0, 0, 'aircraft %s' %(k+1))

for i in range(len(demand)):
    sheetx.write(i+1,0, airports[i])
    sheetw.write(i+1,0, airports[i])

    for k in range(num_ac_types):
        locals()['sheet_{0}'.format(k)].write(i+1, 0, airports[i])

    for j in range(len(demand[i])):
        sheetx.write(0, j+1, airports[j])
        sheetw.write(0, j+1, airports[j])

        sheetx.write(i+1, j+1, newx[i,j])
        sheetw.write(i+1, j+1, neww[i,j])

for k in range(num_ac_types):
    writtenlines= 0
    for r in range(len(routes)): 
        if z[r,k].x > 0:
            routestring = ""
            for routeitem in routes[r]:
                routestring = routestring + str(routeitem) + " - "

            locals()['sheet_{0}'.format(k)].write(writtenlines+1, 1, routestring)
            locals()['sheet_{0}'.format(k)].write(writtenlines+1, 2, z[r,k].x)
            writtenlines= writtenlines + 1

# Saving the Excel file
wb.save('Model_2_results.xls')
print("Wrote results file")

#######################################################################################################################
##############################################   CALCULATING KPIs   ###################################################
#######################################################################################################################

totalcost = 0

for k in range(5):
    totalcost = totalcost + AC[k].x * k_aircraft[k][6]
    for r in range(len(routes)):
        if z[r,k].x > 0:
            totalcost = totalcost + cost[r][k] * z[r,k].x

totalrevenue = totalcost + f_objective

RPK = []

for i in range(len(airports)):
   for j in range(len(airports)):
       rpkvalue = 0
       for r in range(len(routes)):
           if x[i,j,r].x > 0:
               rpkvalue = rpkvalue + x[i,j,r].x * dij[i,j]
       if rpkvalue > 0:
           RPK.append(rpkvalue)