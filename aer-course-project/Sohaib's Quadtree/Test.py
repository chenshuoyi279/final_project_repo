from QuadTreeMap import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

## OFFLINE TESTING OF QUADTREE FOR DEBUG 
########################### Create World

start = [-1.0, -3.0, 0.]
target = [-0.5, 2.0,1.0]
Goal = Point(target,0.025)
gates= [  # x, y, z, r, p, y, type
            [0.5, -2.5, 0, 0, 0, -1.57, 0],  # gate 1
            [2.0, -1.5, 0, 0, 0, 0, 0],  # gate 2
            [0.0, 0.2, 0, 0, 0, 1.57, 0],  # gate 3
            [-0.5, 1.5, 0, 0, 0, 0, 0]  # gate 4
        ]

obstacles = [  # x, y, z, r, p, y
        [1.5, -2.5, 0, 0, 0, 0],  # obstacle 1
        [0.5, -1.0, 0, 0, 0, 0],  # obstacle 2
        [1.5, 0, 0, 0, 0, 0],  # obstacle 3
        [-1.0, 0, 0, 0, 0, 0]  # obstacle 4
    ]
    
#print(len(obstacles))
plt.figure()
plt.xlim([-3,3])
plt.ylim([-3,3])
O = [Obs(obstacles[i],0.2,0.2) for i in range(len(obstacles))]
G = [Gate(gates[i]) for i in range(len(gates))]

for i in range(len(gates)):
    for j in range(len(G[i].obs)):
        O.append(G[i].obs[j])


center = np.array([0.,0.,1.])
size = np.array([6,6,2])
Tree = quadtreeNode(center,size)
Tree.insert(Goal, 'Gate')

print_tree(Tree, False)


for i in range(len(G)):
    Tree.insert(G[i],flag = 'Gate')
    G[i].draw()
for i in range(len(O)):
    Tree.insert(O[i],flag = 'Obs')
    O[i].draw()
print_tree(Tree)
Goal.draw()
print_tree(Tree,True)


pt = start
k = 0


#plt.show()
start = np.array([-1., -3., 1.])
GateSequence = [0,1,2,3]
count = 0
for k in GateSequence:
    awps = Astar(Tree,start,G[k],1)
    #print(k,awps )
    if G[k].isRotated:
        if awps[-1][0] < G[k].cx:
            delta = [0.5, 0., 0.]
        else:
            delta = [-0.5, 0., 0.]
    else:
        if awps[-1][1] < G[k].cy:
            delta = [0.0, 0.5, 0.]
        else:
            delta = [0.0, -0.5, 0.]
    if count == 0:
        AWPS = np.vstack((start, awps))
    else:
        AWPS = np.vstack((AWPS, awps))
    start = awps[-1] + delta
    count +=1

#awps = Astar(Tree,start,Goal,1)
AWPS = np.vstack((AWPS,start,start,start))
plt.plot(AWPS[:,0],AWPS[:,1],'-go',label = 'A* Waypoints')
AWPS_smooth = moving_avg_filter(AWPS,3)
AWPS_smooth = np.vstack((pt,AWPS_smooth))


plt.plot(AWPS_smooth[:,0],AWPS_smooth[:,1],'-bx',label = 'Moving Average Waypoints')

t = np.arange(AWPS_smooth.shape[0])
# Define the x and y values for the spline interpolation
xnew = np.linspace(t.min(), t.max(), 300)
splx = make_interp_spline(t, AWPS_smooth[:,0])
# Evaluate the spline at the new x values
APS_xnew = splx(xnew)
# Define the x and y values for the spline interpolation

sply = make_interp_spline(t, AWPS_smooth[:,1])
# Evaluate the spline at the new x values
APS_ynew = sply(xnew)

plt.plot(APS_xnew,APS_ynew, '-m', label = 'Splined Interpolation')


plt.legend()
plt.show()
