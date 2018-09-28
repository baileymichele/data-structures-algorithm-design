# kdtrees.py
"""Data Structures 3 (K-d Trees).
Bailey Smith
13 October 2016
"""

import random
import numpy as np
from trees import BST
from sklearn import neighbors
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

def metric(x, y):
    """Return the euclidean distance between the 1-D arrays 'x' and 'y'.

    Raises:
        ValueError: if 'x' and 'y' have different lengths.

    Example:
        >>> metric([1,2],[2,2])
        1.0
        >>> metric([1,2,1],[2,2])
        ValueError: Incompatible dimensions.
    """
    if len(x) != len(y):
        raise ValueError("Incompatible dimensions")
    else:
        return euclidean(x, y)


def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively.
    Check the distances between 'target' and each point in 'data_set'.
    Use the Euclidean metric to calculate distances.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        ((k,) ndarray) the member of 'data_set' that is nearest to 'target'.
        (float) The distance from the nearest neighbor to 'target'.
    """
    shortest_dist = 0#Distance to closest point
    for i in xrange(len(data_set)):#len: gives # of rows
        point = data_set[i,:]#Gives Each separate row
        current = metric(target, point)#Distance to current point
        if i == 0 or current < shortest_dist:
            closest = point
            shortest_dist = current
    return closest, shortest_dist

class BSTNode(object):
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value
class KDTNode(BSTNode):
    def __init__(self, data):
        BSTNode.__init__(self, data)
        if type(data) is not np.ndarray:
            raise TypeError("Invalid type of data")
        self.axis = 0

class KDT(BST):
    """A k-dimensional binary search tree object.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a NumPy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """
    def __init__(self):
        BST.__init__(self)

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursion on the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Return the new node. This method should be similar to BST.insert().
        """
        def _step(current,previous):#PASS in PREVIOUS
            """Recursively step through the tree until the parent node for node
            containing 'data' is found."""
            new_node = KDTNode(data)
            if current is None:
                new_node.prev = previous#Looks at the parent node of what we are adding
                if data[previous.axis] < previous.value[previous.axis]:
                    previous.left = new_node
                else:
                    previous.right = new_node
                #...Add..set axis to be 1 more than parent
                #IF AT LAST DIMENSION new_node.axis = 0
                if len(data) - 1 != previous.axis:#not sure if this works
                    new_node.axis = previous.axis + 1
            elif np.allclose(data, current.value):
                raise ValueError(str(data) + " is already in the tree.")
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left, current)
            else:
                return _step(current.right, current)

        if self.root == None:
            new_node = KDTNode(data)
            self.root = new_node
        else:
            _step(self.root,None)

def nearest_neighbor(data_set, target):
    """Use your KDT class to solve the nearest neighbor problem.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        The point in the tree that is nearest to 'target' ((k,) ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    #Insert each row of data_set to KD tree
    my_tree = KDT()
    for i in xrange(len(data_set)):
        my_tree.insert(data_set[i,:])
    def KDTsearch(current, neighbor, distance):
        """The actual nearest neighbor search algorithm.

        Inputs:
            current (KDTNode): the node to examine.
            neighbor (KDTNode): the current nearest neighbor.
            distance (float): the current minimum distance.

        Returns:
            neighbor (KDTNode): The new nearest neighbor in the tree.
            distance (float): the new minimum distance.
        """
        if current is None:
            return neighbor, distance
        index = current.axis
        if metric(current.value, target) < distance:
            neighbor = current
            distance = metric(current.value, target)
        if target[index] < current.value[index]:
            neighbor, distance = KDTsearch(current.left, neighbor, distance)
            if target[index] + distance >= current.value[index]:
                neighbor, distance = KDTsearch(current.right, neighbor, distance)
        else:
            neighbor, distance = KDTsearch(current.right, neighbor, distance)
            if target[index] - distance <= current.value[index]:
                neighbor, distance = KDTsearch(current.left, neighbor, distance)
        return neighbor, distance

    distance = metric(target, my_tree.root.value)
    node, dist = KDTsearch(my_tree.root,my_tree.root,distance)
    return node.value, dist


def postal_problem():
    """
    takes a long time to run

    NOTES
    np.average(prediction/testlabels) DO NOT DO, gets divide by 0 errors
    INSTEAD:
    np.sum(predictions == testlabels)/float(len(predictions)) #percentage of correct predictions
    """
    # The United States Postal Service has made a collection of la- beled hand written
    # digits available to the public, provided in PostalData.npz. We will use this data
    # for k-nearest neighbor classification. This data set may be loaded by using the
    # following command
    # labels, points, testlabels, testpoints = np.load('PostalData.npz').items()

    # 'n_neighbors' determines how many neighbors to give votes to.
    # 'weights' may be 'uniform' or 'distance.' The 'distance' option
    # gives nearer neighbors more weight.
    # 'p=2' instructs the class to use the euclidean metric.
    # nbrs = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance', p=2)

    # 'points' is some NumPy array of data
    # 'labels' is a NumPy array of labels describing the data in points.
    # nbrs.fit(points[1], labels[1])

    # 'testpoints' is an array of unlabeled points.
    #Perform the search and calculate the accuracy of the classification.
    # prediction = nbrs.predict(testpoints)
    # np.sum(predictions == testlabels)/float(len(predictions))#Save in list: append

    labels, points, testlabels, testpoints = np.load('PostalData.npz').items()
    def helper(n, w):
        nbrs = neighbors.KNeighborsClassifier(n_neighbors=n, weights=w, p=2)
        nbrs.fit(points[1], labels[1])
        prediction = nbrs.predict(testpoints[1])
        return np.average(prediction == testlabels[1])
    for j in ["distance","uniform"]:
        for i in [1,4,10]:
            print "n_neighbors =", i, "weight =", j, ":", helper(i,j)



