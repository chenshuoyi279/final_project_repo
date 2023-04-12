import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

min_point = np.array([-3., -3., 0])
max_point = np.array([3., 3., 2])

obs_size = np.array([0.06, 0.06, 1.5])
gate_size = np.array([0.10, 0.40, 0.4])

obs_locs = np.array([[1.5, -2.5, 0.75],
                     [0.5, -1.0, 0.75],
                     [1.5,  0.,  0.75],
                     [-1.,  0.,  0.75]])
gate_locs = np.array([[0.5, -2.5, 1.],
                      [2.,  -1.5, 1.],
                      [0.,   0.2, 1.],
                      [-0.5, 1.5, 1.]])

class OctreeNode:
    def __init__(self, center, size,parent=None,label=None):
        self.center = center
        self.size = size
        if parent is not None:
            self.Depth = parent.Depth +1
        else: self.Depth = 0
        if label is None:
            self.Label = '0'
        else:
            self.Label = parent.Label
            self.Label = self.Label + str(label)
        self.bounds = np.array([center-0.5*size,center+0.5*size])
        self.children = [None] * 8
        self.parent = parent
        self.isLeaf = True
        self.containsObs = False
        self.containsGate = False

        print(f'NewNode C{self.center}')
        print(f'NewNode Sz{self.size}')
        print(f'NewNode Bnd{self.bounds}')

    def check_volume_in_cube(self, volume_center, volume_bnds):
        # Calculate the minimum and maximum bounds of the volume
        volume_min = volume_bnds[0]
        volume_max = volume_bnds[1]

        # Check if the volume is completely within the cube
        if all(volume_min[i] >= self.bounds[0][i] and volume_max[i] <= self.bounds[1][i] for i in range(len(self.bounds[0]))):
            print("Completely within cube")
            return True

        # Check if the volume is somewhat within the cube
        elif any(volume_min[i] <= self.bounds[1][i] and volume_max[i] >= self.bounds[0][i] for i in range(len(self.bounds[0]))):
            print("Somewhat within cube")
            return True

        # Volume is not within cube
        else:
            print("Not within cube")
            return False

    def within_bounds(self,objCenter):

        print((self.bounds[0] <= objCenter))
        print((self.bounds[1] >= objCenter))
        return np.all(self.bounds[0] <= objCenter) and np.all(self.bounds[1] >= objCenter)

    def pt_within_bounds(self,pt):
        return np.all(self.bounds[0] <= pt) and np.all(self.bounds[1] >= pt)

class Octree:
    def __init__(self,OctreeNode):
        self.MaxDepth = 6;
        self.curDepth = 0;
        self.root = OctreeNode;

    def add_obstacle(self,RootNode,ObsLoc,ObsSize):
        ObsBounds = np.array([ObsLoc - ObsSize*0.5, ObsLoc+ ObsSize*0.5])
        #print(ObsLoc)
        #print(ObsSize, ObsSize*0.5)
        print(ObsBounds)
        # Check whether obstacle is within bounds
        if RootNode.check_volume_in_cube(ObsLoc,ObsBounds):
            RootNode.containsObs = True
            if RootNode.isLeaf:
                if self.curDepth < self.MaxDepth:
                    RootNode = self.subdivide(RootNode)
                    RootNode.isLeaf = False
                    for i in range(8):
                        self.add_obstacle(RootNode.children[i],ObsLoc,ObsSize)
            else:
                 self.curDepth +=1
                 for i in range(8):
                    self.add_obstacle(RootNode.children[i],ObsLoc,ObsSize)



        else:
            print(f'obstacle not within bounds of node {RootNode.Label}')
        #    if RootNode.parent is None:

    def subdivide(self,CurNode):
        self.curDepth +=1
        print(f'subdividing')
        new_size = CurNode.size*0.25
        dx,dy,dz = new_size
        print(f'newsize {new_size}')
        print(f'newsize{np.array([ dx, -dy, dz])}')
        print(f'NewNode center: {CurNode.center + np.array([ dx, dy, dz]) }')
        CurNode.children[0] = OctreeNode(CurNode.center + np.array([ dx, dy, dz]) ,2*new_size,parent=CurNode,label =0)
        CurNode.children[1] = OctreeNode(CurNode.center + np.array([ dx,-dy, dz]) ,2*new_size,parent=CurNode,label =1)
        CurNode.children[2] = OctreeNode(CurNode.center + np.array([ dx,-dy,-dz]) ,2*new_size,parent=CurNode,label =2)
        CurNode.children[3] = OctreeNode(CurNode.center + np.array([ dx, dy,-dz]) ,2*new_size,parent=CurNode,label =3)
        CurNode.children[4] = OctreeNode(CurNode.center + np.array([-dx, dy, dz]) ,2*new_size,parent=CurNode,label =4)
        CurNode.children[5] = OctreeNode(CurNode.center + np.array([-dx,-dy, dz]) ,2*new_size,parent=CurNode,label =5)
        CurNode.children[6] = OctreeNode(CurNode.center + np.array([-dx,-dy,-dz]) ,2*new_size,parent=CurNode,label =6)
        CurNode.children[7] = OctreeNode(CurNode.center + np.array([-dx, dy,-dz]) ,2*new_size,parent=CurNode,label =7)
        return CurNode

    def print_tree(self,node, level=0):
        """
        Print all the nodes in the tree and indent each time you go down a level
        """
        if node is not None:
            print(node.Label +'\t' +str(node.center) + '\t'+ str(node.Depth) + '\tObs:\t' + str(node.containsObs) + '\tGate:\t' + str(node.containsGate))
            for child in node.children:
                self.print_tree(child, level + 1)


center = np.array([0.,0.,1.])
size =np.array([6.,6.,2.])
#print(center+0.5*size)
BaseNode = OctreeNode(center,size)
Tree = Octree(BaseNode)
Tree.add_obstacle(Tree.root,obs_locs[0],obs_size)
Tree.curDepth = 0
Tree.print_tree(Tree.root)
print('-----------------------------------------')
#Tree.add_obstacle(Tree.root,obs_locs[2],obs_size)
#Tree.print_tree(Tree.root)

#test.children[0].center