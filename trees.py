# -*- coding: utf-8 -*-
# trees.py
"""Data Structures II (Trees).
Bailey Smith
October 6 2016
"""

import time
import random
import numpy as np
from matplotlib import pyplot as plt

class SinglyLinkedListNode(object):
    """Simple singly linked list node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList(object):
    """A very simple singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None
    def append(self, data):
        """Add a Node containing 'data' to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

def iterative_search(linkedlist, data):
    """Search 'linkedlist' iteratively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    current = linkedlist.head
    while current is not None:
        if current.value == data:
            return current
        current = current.next
    raise ValueError(str(data) + " is not in the list.")

def recursive_search(linkedlist, data):
    """Search 'linkedlist' recursively for a node containing 'data'.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError.

    Inputs:
        linkedlist (SinglyLinkedList): a linked list object.
        data: the data to search for in the list.

    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    if linkedlist.head == None:
        raise ValueError("Can't find the node because list is empty or there is no such node in the list")
        #If there is no such node in the list raise a ValueError.
    elif linkedlist.head.value == data:
        return linkedlist.head
    else:
        linkedlist.head = linkedlist.head.next
        return recursive_search(linkedlist, data)



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


class BST(object):
    """Binary Search Tree data structure class.
    The 'root' attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

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
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Step to the left.
                return _step(current.left)
            else:                                   # Step to the right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.

        Example:
            >>> b = BST()       |   >>> b.insert(1)     |       (4)
            >>> b.insert(4)     |   >>> print(b)        |       / \
            >>> b.insert(3)     |   [4]                 |     (3) (6)
            >>> b.insert(6)     |   [3, 6]              |     /   / \
            >>> b.insert(5)     |   [1, 5, 7]           |   (1) (5) (7)
            >>> b.insert(7)     |   [8]                 |             \
            >>> b.insert(8)     |                       |             (8)
        """
        def _step(current,previous):#pass in previous so when we get to the dead end we have access to parent
            """Recursively step through the tree until the node containing
            'data' is found. If it is found, raise a Value Error because we
            don't want duplicates. if we reach a "dead end" that is where we
            insert.
            """
            new_node = BSTNode(data)
            if current is None:# Base case 1: dead end. add node to tree
                new_node.prev = previous
                if data < previous.value:
                    previous.left = new_node
                else:
                    previous.right = new_node
            elif data == current.value:               # Base case 2: data found!
                raise ValueError(str(data) + " is already in the tree.")
            elif data < current.value:                # Step to the left.
                return _step(current.left,current)
            else:                                   # Step to the right.
                return _step(current.right,current)

        if self.root == None:
            new_node = BSTNode(data)
            self.root = new_node
        else:
            _step(self.root,None)

    def inorder(self, temp1, temp2): #pass in the node that is needing to be removed(temp1) and it's right node(temp2)
        if temp2.left == None:
            temp1.value = temp2.value
            if temp2 is temp1.right:
                temp1.right = None#delete temp2;
            else:
                temp2.prev.left = None

    	elif temp2.left is not None:
    		self.inorder(temp1, temp2.left)


    def remove(self, data):
        """
            If the tree is empty, or if there is no node containing 'data',
            raises a ValueError.

        Examples:
            >>> print(b)        |   >>> b.remove(1)     |   [5]
            [4]                 |   >>> b.remove(7)     |   [3, 8]
            [3, 6]              |   >>> b.remove(6)     |
            [1, 5, 7]           |   >>> b.remove(4)     |
            [8]                 |   >>> print(b)        |
        """
        to_remove = self.find(data)

        '''1. The tree is empty
            How do I use find to take care of these cases'''
        if self.root == None:
            raise ValueError("Cannot remove because the tree is empty")
        #'''2. The target is the root:'''
        elif to_remove.value == self.root.value:
            #'''a. The root is a leaf node, hence the only node in the tree'''
            if self.root.left == None and self.root.right == None:
                self.root = None
            #'''b. The root has one child'''
            elif self.root.left == None:
                self.root.right.prev = None
                self.root = self.root.right
            elif self.root.right == None:
                self.root.left.prev = None
                self.root = self.root.left
            #'''c. The root has two children'''
            elif self.root.left is not None and self.root.right is not None:
                self.inorder(to_remove, to_remove.right)

        #'''3. The target is not the root:
            #a. The target is a leaf node'''
        elif to_remove.left == None and to_remove.right == None:
            if to_remove.prev.left is to_remove:
                to_remove.prev.left = None
            elif to_remove.prev.right is to_remove:
                to_remove.prev.right = None
        # '''3. The target is not the root:
        #     b. The target has one child'''
        elif to_remove.left == None:
            to_remove.right.prev = to_remove.prev
            if to_remove.prev.left is to_remove:
                to_remove.prev.left = to_remove.right
            elif to_remove.prev.right is to_remove:
                to_remove.prev.right = to_remove.right
        elif to_remove.right == None:
            to_remove.left.prev = to_remove.prev
            if to_remove.prev.left is to_remove:
                to_remove.prev.left = to_remove.left
            elif to_remove.prev.right is to_remove:
                to_remove.prev.right = to_remove.left
        # '''3. The target is not the root:
        #     c. The target has two children'''
        else:
            self.inorder(to_remove, to_remove.right)

    def __str__(self):
        """String representation: a hierarchical view of the BST.
        Do not modify this method, but use it often to test this class.
        (this method uses a depth-first search; can you explain how?)

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed out
                (2) (5)    [2, 5]       by depth levels. The edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """

        if self.root is None:
            return "[]"
        str_tree = [list() for i in xrange(_height(self.root) + 1)]
        visited = set()

        def _visit(current, depth):
            """Add the data contained in 'current' to its proper depth level
            list and mark as visited. Continue recusively until all nodes have
            been visited.
            """
            str_tree[depth].append(current.value)
            visited.add(current)
            if current.left and current.left not in visited:
                _visit(current.left, depth+1)
            if current.right and current.right not in visited:
                _visit(current.right, depth+1)

        _visit(self.root, 0)
        out = ""
        for level in str_tree:
            if level != list():
                out += str(level) + "\n"
            else:
                break
        return out


class AVL(BST):
    """AVL Binary Search Tree data structure class. Inherits from the BST
    class. Includes methods for rebalancing upon insertion. If your
    BST.insert() method works correctly, this class will work correctly.
    Do not modify.
    """
    def _checkBalance(self, n):
        return abs(_height(n.left) - _height(n.right)) >= 2

    def _rotateLeftLeft(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateRightRight(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp

    def _rotateLeftRight(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotateLeftLeft(n)

    def _rotateRightLeft(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotateRightRight(n)

    def _rebalance(self,n):
        """Rebalance the subtree starting at the node 'n'."""
        if self._checkBalance(n):
            if _height(n.left) > _height(n.right):
                if _height(n.left.left) > _height(n.left.right):
                    n = self._rotateLeftLeft(n)
                else:
                    n = self._rotateLeftRight(n)
            else:
                if _height(n.right.right) > _height(n.right.left):
                    n = self._rotateRightRight(n)
                else:
                    n = self._rotateRightLeft(n)
        return n

    def insert(self, data):
        """Insert a node containing 'data' into the tree, then rebalance."""
        BST.insert(self, data)
        n = self.find(data)
        while n:
            n = self._rebalance(n)
            n = n.prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")

def _height(current):
    """Calculates the height of a given node by descending recursively until
    there are no further child nodes. Returns the number of children in the
    longest chain down.

    This is a helper function for the AVL class and BST.__str__().
    Abandon hope all ye who modify this function.

                                node | height
    Example:  (c)                  a | 0
              / \                  b | 1
            (b) (f)                c | 3
            /   / \                d | 1
          (a) (d) (g)              e | 0
                \                  f | 2
                (e)                g | 0
    """
    if current is None:
        return -1
    return 1 + max(_height(current.right), _height(current.left))


def time_comparison():
    """Compare the build and search speeds of the SinglyLinkedList, BST, and
    AVL classes. For search times, use iterative_search(), BST.find(), and
    AVL.find() to search for 5 random elements in each structure. Plot the
    number of elements in the structure versus the build and search times.
    Use log scales if appropriate.
    """
    data = []
    loadLL = []
    findLL = []
    loadBST = []
    findBST = []
    loadAVL = []
    findAVL = []


    with open("english.txt", 'r') as myfile:
        for line in myfile:
            data.append(line.strip())

    #SET DOMAIN
    domain = 2**np.arange(3,12)
    domain1 = []
    domain2 = []
    for n in domain:
        my_LL = SinglyLinkedList()
        my_BST = BST()
        my_AVL = AVL()
        domain1 = random.sample(data,n)
        """Timing load into SinglyLinkedList"""
        start = time.time()
        for i in domain1:
            my_LL.append(i)
        loadLL.append(time.time() - start)#Putting times into a list so we can plot
        """Timing load into BST"""
        start = time.time()
        for i in domain1:
            my_BST.insert(i)
        loadBST.append(time.time() - start)
        """Timing load into AVL"""
        start = time.time()
        for i in domain1:
            my_AVL.insert(i)
        loadAVL.append(time.time() - start)

        domain2 = random.sample(domain1,5)
        """Timing find in SinglyLinkedList"""
        start = time.time()
        for j in domain2:
            iterative_search(my_LL, j)
        findLL.append(time.time() - start)
        """Timing find in BST"""
        start = time.time()
        for j in domain2:
            my_BST.find(j)
        findBST.append(time.time() - start)
        """Timing find in AVL"""
        start = time.time()
        for j in domain2:
            my_AVL.find(j)
        findAVL.append(time.time() - start)

    #LOG scale?
    plt.subplot(121)
    plt.loglog(domain, loadLL, 'b.-', basex=2, basey=2, lw=2, ms=12, label="Singly Linked List")
    plt.loglog(domain, loadBST, 'g.-', basex=2, basey=2, lw=2, ms=12, label="Binary Search Tree")
    plt.loglog(domain, loadAVL, 'r.-', basex=2, basey=2, lw=2, ms=12, label="AVL Tree")
    plt.title("Build Times", fontsize=15)
    plt.legend(loc="upper left")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")

    plt.subplot(122)
    plt.loglog(domain, findLL, 'b.-', basex=2, basey=2, lw=2, ms=12, label="SinglyLinkedList")
    plt.loglog(domain, findBST, 'g.-', basex=2, basey=2, lw=2, ms=12, label="Binary Search Tree")
    plt.loglog(domain, findAVL, 'r.-', basex=2, basey=2, lw=2, ms=12, label="AVL Tree")
    plt.title("Search Times", fontsize=15)
    plt.legend(loc="upper left")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")

    plt.show()


if __name__ == '__main__':
    my_list = SinglyLinkedList()
    my_list.append("hello")
    my_list.append(1)
    my_list.append("yay")
    print iterative_search(my_list, 1)
    print recursive_search(my_list, "yay")
    
    my_bst = BST()
    my_bst.insert(4)
    my_bst.insert(6)
    my_bst.insert(3)
    my_bst.remove(4)
    my_bst.insert(5)
    my_bst.insert(7)
    my_bst.insert(8)
    my_bst.insert(1)
    print my_bst
    print "\n"
    my_bst.insert(4)
    my_bst.remove(1)
    my_bst.remove(7)
    my_bst.remove(6)
    my_bst.remove(4)
    print my_bst

    print time_comparison()
