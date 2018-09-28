# -*- coding: utf-8 -*-
# linked_lists.py
"""Data Structures 1 (Linked Lists)
Bailey Smith
September 29 2016
"""

import types

class Node(object):
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store 'data' in the 'value' attribute if it is a string, int, float, or long."""
        if isinstance(data,types.StringType) or isinstance(data,types.IntType) or isinstance(data,types.LongType) or isinstance(data,types.FloatType):
            self.value = data
        else:
            raise TypeError("The data must be a string, int, float, or long")


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store 'data' in the 'value' attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None
        self.prev = None


class LinkedList(object):
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the 'head' and 'tail' attributes by setting
        them to 'None', since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.count = 0#attribute to keep track of length of list

    def append(self, data):
        """Append a new node containing 'data' to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.count += 1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.count += 1

    def find(self, data):
        """Return the first node in the list containing 'data'.
        If no such node exists, raise a ValueError.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5,7,9]:
            ...     l.append(i)
            ...
            >>> node = l.find(5)
            >>> node.value
            5
            >>> l.find(10)
            ValueError: <message>
        """
        if self.head is None:
            raise ValueError("The list is empty")
        else:
            current = self.head
            while not current is self.tail:
                if current.value == data:
                    return current
                else:
                    current = current.next
                    if current.value == data:#Otherwise if data is in tail it won't be checked?
                        return current
            raise ValueError("No such node exists")

    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5]:
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        """Make count an attribute..but where? I'm guessing in LinkedList init: then just return self.count"""
        return self.count


    def __str__(self):#self is a linked list
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()   |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:   |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)    |   ...     l2.append(i)
            ...                     |   ...
            >>> print(l1)           |   >>> print(l2)
            [1, 3, 5]               |   ['a', 'b', 'c']
        """
        """If the data is a number do 1 thing if data is string do something else"""
        current = self.head
        my_list = []
        for i in xrange(len(self)):
            my_list.append(current.value)
            current = current.next
        return str(my_list)

    def remove(self, data):
        """Remove the first node in the list containing 'data'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'data'.

        Examples:
            >>> print(l1)       |   >>> print(l2)
            [1, 3, 5, 7, 9]     |   [2, 4, 6, 8]
            >>> l1.remove(5)    |   >>> l2.remove(10)
            >>> l1.remove(1)    |   ValueError: <message>
            >>> l1.remove(9)    |   >>> l3 = LinkedList()
            >>> print(l1)       |   >>> l3.remove(10)
            [3, 7]              |   ValueError: <message>
        """
        """Don't forget to decrement 'count'"""
        to_remove = self.find(data)
        if self.head == None:
            raise ValueError("Can't remove because list is empty")
        elif self.head is self.tail:#If the one I want is only node in list (find function accounted for if what I want is not in list)
            self.head = None
            self.tail = None
            self.count = 0
        elif to_remove.value == self.tail.value:#if removing tail
            to_remove.prev.next = None
            self.tail = to_remove.prev
            self.count -= 1
        elif to_remove.value == self.head.value:#If removing head
            to_remove.next.prev = None
            self.head = to_remove.next
            self.count -= 1
        else:
            to_remove.prev.next = to_remove.next
            to_remove.next.prev = to_remove.prev
            self.count -= 1

    def insert(self, data, place):
        """Insert a node containing 'data' immediately before the first node
        in the list containing 'place'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'place'.

        Examples:
            >>> print(l1)           |   >>> print(l1)
            [1, 3, 7]               |   [1, 3, 5, 7, 7]
            >>> l1.insert(7,7)      |   >>> l1.insert(3, 2)
            >>> print(l1)           |   ValueError: <message>
            [1, 3, 7, 7]            |
            >>> l1.insert(5,7)      |   >>> l2 = LinkedList()
            >>> print(l1)           |   >>> l2.insert(10,10)
            [1, 3, 5, 7, 7]         |   ValueError: <message>
        """
        """Don't have to do anything different if self.tail is place b/c we insert befere place"""
        where_to_insert = self.find(place)
        new_node = Node(data)
        if self.head is where_to_insert:#Insert at head
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self.count += 1
        else:
            where_to_insert.prev.next = new_node
            new_node.prev = where_to_insert.prev.next
            new_node.next = where_to_insert
            where_to_insert.prev = new_node
            self.count += 1

class Deque(LinkedList):

    def __init__(self):
        LinkedList.__init__(self)

    def pop(self):
        if self.head == None:
            raise ValueError("Can't remove because list is empty")
        elif self.head is self.tail:#If only one node in list
            data = self.head.value
            self.head = None
            self.tail = None
            self.count = 0
        else:
            to_remove = self.tail
            data = to_remove.value
            to_remove.prev.next = None
            self.tail = to_remove.prev
            self.count -= 1
        return data

    def popleft(self):
        if self.head == None:
            raise ValueError("Can't remove because list is empty")
        elif self.head is self.tail:#If only one node in list
            data = self.head.value
            self.head = None
            self.tail = None
            self.count = 0
        else:
            to_remove = self.head
            data = to_remove.value
            to_remove.next.prev = None
            self.head = to_remove.next
            self.count -= 1
        return data

    def appendleft(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node
        self.count += 1

    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")
    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() for inserting")

def reverse(infile, outfile):
    """Reverse the file 'infile' by line and write the results to 'outfile'."""

    """Append each line in file to Deque
    Pop each entry from Deque and write the entry to new file"""
    lines = Deque()
    with open(infile, 'r') as myfile:
        for line in myfile:
            lines.append(line.strip())

    with open(outfile, 'w') as outfile:
        lines_size = len(lines) #need to keep track of deque size before popping
        for i in xrange(lines_size):
            my_line = lines.pop()
            if i < lines_size - 1:
                outfile.write(my_line)
                outfile.write("\n")
            else:
                outfile.write(my_line)

