#~*~ coding: UTF-8 ~*~
# oop.py
"""Object Oriented Programming.
Bailey Smith
September 15, 2016
"""

class Backpack(object):
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): color of the backpack
        max_size (int): maximum ammount of contents
    """

    def __init__(self, name, color, max_size=5):
        """Set the name and initialize an empty contents list.

        Inputs:
            name (str): the name of the backpack's owner.

        Returns:
            A Backpack object wth no contents.
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size

    def put(self, item):
        """Add 'item' to the backpack's list of contents."""
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print "No Room!"

    def take(self, item):
        """Remove 'item' from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Resets the contents of the back- pack to an empty list."""
        self.contents = []

    # Magic Methods ----------------------------------------------------------

    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    def __eq__(self, other):
        if self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents):
            equal = True
        else:
            equal = False
        return equal

    def __str__(self):
        owner = "Owner:\t\t" + self.name
        color = "\nColor:\t\t" + self.color
        size = "\nSize:\t\t" + str(len(self.contents))
        max_size = "\nMax Size:\t" + str(self.max_size)
        contents =  "\nContents\t" + str(self.contents)
        string = owner + color + size + max_size + contents
        return string

class Jetpack(Backpack):
    def __init__(self, name, color,max_size=2,fuel=10):
        Backpack.__init__(self, name, color, max_size)
        self.max_size = max_size
        self.fuel = fuel

    def fly(self, ammount):
        if ammount <= self.fuel:
            self.fuel -= ammount
        else:
            print "Not enough fuel!"

    def dump(self):
        Backpack.dump(self)
        self.fuel = 0

'''Testing Backpack class'''
def test_backpack():
    testpack = Backpack("Bailey", "black and white")       # Instantiate the object.
    if testpack.max_size != 5:                  # Test an attribute.
        print("Wrong default max_size!")
    for item in ["pencil", "pen", "paper", "computer", "textbook"]:
        testpack.put(item)                      # Test a method.
    print(testpack.contents)
    testpack.put("water bottle")
    testpack.take("pencil")
    print(testpack.contents)
    testpack.dump()
    print(testpack.contents)
    print (testpack)

'''Testing Jetpack class'''
def test_jetpack():
    testpack = Jetpack("Brooke", "Rainbow")
    if testpack.fuel != 10:
        print("Wrong default fuel")
    testpack.put("glasses")
    print(testpack.contents)
    testpack.fly(8)
    print(testpack.fuel)
    testpack.fly(3)
    testpack.dump()
    print(testpack.contents)
    print(testpack.fuel)

def test_complex():
    cn1 = ComplexNumber(1,-1)
    cn2 = ComplexNumber(2,1)
    print(cn1/cn2)

class ComplexNumber(object):

    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def conjugate(self):
        '''Returns comlex conjugate as a new ComplexNumber object'''
        return ComplexNumber(self.real,self.imag * -1)
# Magic Methods ---------------------------------------------------------------

    def __abs__(self):
        return (self.real**2 + self.imag**2)**.5

    def __lt__(self,other):
        return abs(self) < abs(other)

    def __gt__(self,other):
        return abs(self) > abs(other)

    def __eq__(self,other):
        if self.real == other.real and self.imag == other.imag:
            equal = True
        else:
            equal = False
        return equal
    def __ne__(self,other):
        return not self == other

    def __add__(self,other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real,imag)

    def __sub__(self,other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real,imag)

    def __mul__(self,other):
        real = self.real * other.real - self.imag * other.imag
        imag = self.imag * other.real + self.real * other.imag
        return ComplexNumber(real,imag)

    def __div__(self,other):
        conjugate = other.conjugate()
        real_numerator = self.real*other.imag + self.imag*other.real
        imag_numerator = self.imag*other.real-self.real*other.imag
        denominator = abs(other*conjugate)
        return ComplexNumber(real_numerator/denominator,imag_numerator/denominator)

# if __name__ == "__main__":
    print test_backpack()
    print test_jetpack()
    print test_complex()
