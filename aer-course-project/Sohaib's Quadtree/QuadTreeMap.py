import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import BSpline, splrep

MAX_DEPTH = 6
class Obs():
    def __init__(self,xyzrpyt,dx,dy):
        """ Constructs Obstacle given a dx,dy size"""
        # records center and size of obstacle
        self.cx = xyzrpyt[0]
        self.cy = xyzrpyt[1]
        self.dx = dx
        self.dy = dy
        # calculates bounds of obstacle
        self.maxX = self.cx + self.dx*0.5
        self.maxY = self.cy + self.dy*0.5
        self.minX = self.cx - self.dx*0.5
        self.minY = self.cy - self.dy*0.5

    def draw(self):
        """ Draws Obstacle red on an open graph"""
        plt.gca().add_patch(Rectangle((self.minX,self.minY),
                                       self.dx, self.dy,
                                       facecolor ='red'))
class Point():
    def __init__(self,xyz,dx):
        """ Constructs a point given a square radius / used for the goal point"""
        # records center and size of obstacle
        self.cx = xyz[0]
        self.cy = xyz[1]
        self.dx = dx
        # calculates bounds of obstacle
        self.maxX = self.cx + self.dx
        self.maxY = self.cy + self.dx
        self.minX = self.cx - self.dx
        self.minY = self.cy - self.dx
        # initializes whether point is near gate or obstacle
        self.containsObs = False
        self.containsGate = False
    def draw(self):
        """ Draws purple point with the buffer on an open graph"""
        plt.gca().add_patch(Circle((self.cx, self.cy),
                                    self.dx,
                                    facecolor ='purple'))
class Gate():
    def __init__(self,xyzrpyt):
        # records center of gate from nominal infor
        self.cx = xyzrpyt[0] -0.05
        self.cy = xyzrpyt[1] -0.05
        # checks to see whether the gate is rotated
        # given its orientation, set gate size in x and y axis
        if np.abs(xyzrpyt[5]) == 1.57:
            # exagerated obstacles
            self.dx = 0.5
            self.dy = 0.45
            self.isRotated = True
        else:
            self.dx = 0.45
            self.dy = 0.5
            self.isRotated = False
        # calculate bounds
        self.maxX = self.cx + self.dx*0.5
        self.maxY = self.cy + self.dy*0.5
        self.minX = self.cx - self.dx*0.5
        self.minY = self.cy - self.dy*0.5
        # Add obstacles with a given size to
        if self.isRotated:
            self.obs = [Obs([self.cx, self.maxY],0.5, 0.15), Obs([self.cx,self.minY],0.5,0.15)]
        else:
            self.obs = [Obs([self.maxX, self.cy],0.15, 0.5), Obs([self.minX,self.cy],0.15,0.5)]

    def draw(self):
        """ Draws Green gate on an open graph"""
        plt.gca().add_patch(Rectangle((self.minX,self.minY),
                                       self.dx, self.dy,
                                       facecolor ='green'))


class quadtreeNode():
    def __init__(self, center,size,parent=None,quadrant = 0,label=None):
        """ Constructs a quadtreenode given a c enter and size
            Inputs:
                - center [cx,cy]
                - size [dx,dy]
                - parent, if none assumed root
                - quadrant int (0 = NW, 1 = NE, 2 = SW, 3 = SE)
                - label based on quadrant, if none assumed root, otherwise concatenates with
                    each depth: ex: 013 is the node that is the SE quadrant in the NE quadrant
            """
        # records center and size
        self.cx = center[0]
        self.cy = center[1]
        self.dx = size[0]
        self.dy = size[1]
        # calculate bounds
        self.maxX = self.cx  + self.dx*0.5
        self.maxY = self.cy  + self.dx*0.5
        self.minX = self.cx  - self.dx*0.5
        self.minY = self.cy  - self.dy*0.5
        # initializes children to be none, current node is assumed to be Leaf
        self.children = [None]*4
        self.NE = [None]  # Quad 0 child
        self.NW = [None]  # Quad 1 child
        self.SE = [None]  # Quad 2 child
        self.SW = [None]  # Quad 3 child
        self.isLeaf = True
        self.quadrant = quadrant
        # update depth and label
        if parent is None: # if Root
            self.Label = '0'
            self.Depth = 0
        else: # otherwise update its depth and label
            self.Depth = parent.Depth+1
            self.Label = parent.Label + str(label)
        # record parent
        self.parent = parent
        # initialize node assuming it doesn't contain any obstacles or gates
        self.containsObs = False
        self.containsGate = False
    def subdivide(self):
        """ Subdivides current node into 4 """
        if self.Depth < MAX_DEPTH: # if max depth isn't reached
            # Get new size of child nodes (half of current nodes)
            dx = self.dx/2
            dy = self.dy/2
            size = [dx,dy]
            # get current nodes center
            x = self.cx
            y = self.cy
            # Create new nodes for each child's quadrant
            self.NW = quadtreeNode([x-0.5*dx, y+0.5*dy], size, self, quadrant = 0, label='0')
            self.NE = quadtreeNode([x+0.5*dx, y+0.5*dy], size, self, quadrant = 1, label='1')
            self.SW = quadtreeNode([x-0.5*dx, y-0.5*dy], size, self, quadrant = 2, label='2')
            self.SE = quadtreeNode([x+0.5*dx, y-0.5*dy], size, self, quadrant = 3, label='3')
            # Place nodes in the List
            self.children[0] = self.NW
            self.children[1] = self.NE
            self.children[2] = self.SW
            self.children[3] = self.SE
            # update current nodes isLeaf property
            self.isLeaf = False
    def intersect(self,obj):
        """Check if current node intersects with a given object"""
        xmin = max(self.minX,obj.minX) # calculate maximum x between node's and obj's minimum X
        ymin = max(self.minY,obj.minY) # calculate maximum y between node's and obj's minimum Y
        xmax = min(self.maxX,obj.maxX) # calculate minimum x between node's and obj's maximum X
        ymax = min(self.maxY,obj.maxY) # calculate minimum y between node's and obj's maximum Y

        # if max of the mins is maller than min of maxs, then there's intersection
        test = xmin < xmax and ymin < ymax

        if test :
            return True
        else:
            return False

    def insert(self,Obj, flag ):
        """Insert an obstacle into tree and recursively refine the tree"""
        # Check if object intersects current node
        if self.intersect(Obj):
            if flag == 'Obs':
                self.containsObs = True
            if flag == 'Gate':
                self.containsGate = True

            # Check if max depth has been reached
            if self.Depth < MAX_DEPTH:
                # Check if current node is leaf
                if self.isLeaf:
                    # subdivide node to refine tree
                    self.subdivide()
                # if not leaf, then insert obstacle into children nodes recursively
                for child in self.children:
                    child.insert(Obj, flag)

    def draw(self):
        """Draws a the current nodes as a rectangle on a given open figure"""
        facecolor = 'None'
        alpha = 1
        if self.containsObs and self.isLeaf:
            facecolor = 'red'
            alpha = 0.25
        if self.containsGate and self.isLeaf:
            facecolor = 'green'
            alpha = 0.25
        if self.containsObs and self.containsGate and self.isLeaf:
            facecolor = 'blue'
            alpha = 0.25

        plt.gca().add_patch(Rectangle((self.minX,self.minY),
                                       self.dx, self.dy,
                                       edgecolor ='black',
                                       facecolor= facecolor,
                                       alpha = alpha))
    def query_pts(self,point,X):
        '''
        returns leaf node that contains point
        '''
        if self.intersect(Point(point,0.01)):
            if self.isLeaf:
                X.append(self)
                return X
            else:
                for child in self.children:
                    X.append(child.query_pts(point,X))
        if not all(v is None for v in X):
            return next(item for item in X if item is not None)


    def find_neighbours(self,dir):
        """find neighbours calls two sub functions given a direction
        returns a list of neighbours for a given direction"""
        # Dir: 0 = N ,1 = S ,2 =W ,3 = E E
        neighbours = self.getlargerNeigbors(dir)
        neighbours = self.getsmallerNeighbors(neighbours,dir)
        return neighbours
    def getlargerNeigbors(self,dir):
        """Finds current nodes neighbors that are larger or same size"""
        if dir == 0:  # N
            if self.parent is None:
                return None
            if self.parent.SW == self: # Current node is SW
                return self.parent.NW  # Return node is NW
            if self.parent.SE == self: # Current node is SE
                return self.parent.NE  # Return node is NE
            # if none of the above, find parents northern neighbors
            node = self.parent.getlargerNeigbors(dir)

            if node is None or node.isLeaf:
                return node

            return (node.SW # return SW
                    if self.parent.NW == self # Current node is NW
                    else node.SE) # return SE

        elif dir == 1: # S
            if self.parent is None:
                return None
            if self.parent.NW== self: # Current node is NW
                return self.parent.SW  # Return node is SW
            if self.parent.NE == self: # Current node is NE
                return self.parent.SE  # Return node is SE

            node = self.parent.getlargerNeigbors(dir)
            if node is None or node.isLeaf:
                return node

            return (node.NW # Return NW
                    if self.parent.SW == self # Current Node is SW
                    else node.NE) # Return NE

        elif dir == 2: # W
            if self.parent is None:
                return None
            if self.parent.NE == self:# Current node is NE
                return self.parent.NW # Return node is NW
            if self.parent.SE == self:# Current node is SE
                return self.parent.SW # Return node is SW

            node = self.parent.getlargerNeigbors(dir)
            if node is None or node.isLeaf:
                return node

            return (node.SE       # Return node is SE
                    if self.parent.SW == self # Current node is SW
                    else node.NE) # Return node is NE

        elif dir == 3: # E
            if self.parent is None:
                return None
            if self.parent.NW == self: # Current node is NW
                return self.parent.NE  # Return node is NE
            if self.parent.SW == self: # Current node is SW
                return self.parent.SE # Return node is SE

            node = self.parent.getlargerNeigbors(dir)
            if node is None or node.isLeaf:
                return node

            return (node.SW         # Return node is SW
                    if self.parent.SE == self # Current is SE
                    else node.NW)    # Return node is NW

    def getsmallerNeighbors(self,Neigh,dir):
        candidates = [] if Neigh is None else [Neigh]
        neighbors = []
        if dir == 0: # N
            while len(candidates) > 0:
                if candidates[0].isLeaf:
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].SW)
                    candidates.append(candidates[0].SE)
                candidates.remove(candidates[0])
            return neighbors
        if dir == 1: # S
            while len(candidates) > 0:
                if candidates[0].isLeaf:
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].NW)
                    candidates.append(candidates[0].NE)
                candidates.remove(candidates[0])
            return neighbors
        if dir == 2: # W
            while len(candidates) > 0:
                if candidates[0].isLeaf:
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].NE)
                    candidates.append(candidates[0].SE)
                candidates.remove(candidates[0])
            return neighbors
        if dir == 3: # E
            while len(candidates) > 0:
                if candidates[0].isLeaf:
                    neighbors.append(candidates[0])
                else:
                    candidates.append(candidates[0].NW)
                    candidates.append(candidates[0].SW)
                candidates.remove(candidates[0])
            return neighbors
def print_tree(node,verbose = False):
    """ draws tree onto any current open plot
        if verbose is true: tree is printed in terminal with appropriate indents for tree depth
    """
    if verbose:
        print("\t" * node.Depth, f'Node {node.Label}  \t MinXY {node.minX, node.minY} \t dx; {node.dx} \t Depth {node.Depth}'
                             f'\t Obs {node.containsObs} \t Gate {node.containsGate} ')
    for child in node.children:
        if child is not None:
            node = child
            child.draw()
            print_tree(node,verbose)

def moving_avg_filter(path, window_size):
    """Smooths a path using a moving average filter with the specified window size."""
    if len(path) < window_size:
        return path
    # calculate weights to achieve an average
    weights = np.repeat(1.0, window_size) / window_size
    # convolve the weights with the path's x-axis
    smoothed_path = np.convolve(path[:,0], weights, 'valid')
    # convolve on the path's y axis and stack on smoothed path
    smoothed_path = np.vstack((smoothed_path, np.convolve(path[:,1], weights, 'valid')))
    # Add Z = 1 to all waypoints generated by stacking a vector of ones
    smoothed_path = np.vstack((smoothed_path, np.ones((1,smoothed_path.shape[1]))))
    smoothed_path = np.transpose(smoothed_path)

    return smoothed_path

def Astar(Tree,Start,Goal,height):
    """implements A* algorithm to a QuadTree """

    X = [] # initialize empty array for query_point method
    # find current for which the Start point is currently in
    StartNode = Tree.query_pts(Start, X)

    # Initialize Cost-to-go (Manhattan distance) dictionary from start
    g = {}
    g[StartNode.Label] = 0

    # initialize open and closed sets
    open_list= set([StartNode])
    closed_list = set([])
    # initialize a dictionary to keep track of parents
    parents = {}
    parents[StartNode.Label] = StartNode

    # start the algorithm
    while len(open_list) > 0:
        # init a current empty node
        n = None
        # update current node for nodes in the open list
        for K in open_list:
            if n == None or g[K.Label] + manh_d(K,Goal) < g[n.Label] + manh_d(n,Goal):
                n = K
        #  if current node is still empty, means no path was found
        if n == None:
            return None

        # Stopping criteria
        if n.containsGate and not n.containsObs and manh_d(n,Goal) <0.30:
            reconst_path = [] # initialize empty array to reconstruct path
            # traceback the path using the parents of current node
            while parents[n.Label].Label != n.Label:
                reconst_path.append([n.cx,n.cy,height])
                # update last node to be current node's parent
                n = parents[n.Label]
            # add start point to the path
            reconst_path.append([StartNode.cx,StartNode.cy,height])
            reconst_path.reverse()
            return np.array(reconst_path)

        neighbors = []
        # find neighbours in the 4 orthogonal directions
        for dir in range(4):
            neighbors.append(n.find_neighbours(dir))
        # Flatten the list of neighbours
        Neighbours = [Nbr for dir_list in neighbors for Nbr in dir_list]

        for Nbrs in Neighbours:
            # calculate distance from current neighbour to next neighbour
            d = manh_d(Nbrs,Goal)
            if Nbrs not in open_list and Nbrs not in closed_list:
                # if current neighbour isn't in the open and closed sets
                open_list.add(Nbrs)
                parents[Nbrs.Label] = n
                g[Nbrs.Label] = g[n.Label] + d
            else:
                # if current neighbour is in the open set
                if g[Nbrs.Label] > g[n.Label] + d:
                    # check if its listed cost-to-go is greater than
                    # the cost-to-go from the current node
                    g[Nbrs.Label] = g[n.Label] + d
                    parents[Nbrs.Label] = n
                    if Nbrs in closed_list:
                        # update closed and open lists
                        closed_list.remove(Nbrs)
                        open_list.add(Nbrs)
        # once all the neighbors are checked for the current node
        # update open and closed list
        open_list.remove(n)
        closed_list.add(n)


def manh_d(start_node,goal_node):
    """calculates Manhattan distance for current node to goal node"""
    if start_node is None:
        return None
    elif start_node.containsObs:
        # if there's an obstacle within the current node, set Dist = infinity
        return np.inf
    else:
        # manhattan distance
        return  np.abs(start_node.cx - goal_node.cx) +\
                np.abs(start_node.cy - goal_node.cy)