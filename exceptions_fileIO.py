#~*~ coding: UTF-8 ~*~
# exceptions_fileIO.py
"""Introductory Labs: Exceptions and File I/O.
Bailey Smith
22 September 2016
"""

from random import choice
import numpy as np

def arithmagic():
    step_1 = raw_input("Enter a 3-digit number where the first and last "
                                            "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("The first number should be a three digit number.")
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("The first and last digits must differ by 2 or more")

    step_2 = raw_input("Enter the reverse of the first number, obtained "
                                            "by reading it backwards: ")
    if step_1[0] != step_2[2] or step_1[1] != step_2[1] or step_1[2] != step_2[0]:
        raise ValueError("The second number needs to be the reverse of the first number")

    step_3 = raw_input("Enter the positive difference of these numbers: ")
    if abs(int(step_1) - int(step_2)) != int(step_3):
        raise ValueError("This number needs to be the positive difference of the first 2 numbers")

    step_4 = raw_input("Enter the reverse of the previous result: ")
    if step_3[0] != step_4[2] or step_3[1] != step_4[1] or step_3[2] != step_4[0]:#"""DONT KNOW HOW LONG THIS NUMBER IS"""
        raise ValueError("The fourth number needs to be the reverse of the third number")

    print str(step_3) + " + " + str(step_4) + " = 1089 (ta-da!)"


def random_walk(max_iters=1e12):
    try:
        walk = 0
        direction = [-1, 1]
        for i in xrange(int(max_iters)):
            walk += choice(direction)
    except KeyboardInterrupt as KI:
        print "Process interrupted at iteration", i
    else:
        print "Process completed"
    finally:
        return walk

class ContentFilter(object):

    def __init__(self, filename):
        try:
            if type(filename) != str:
                raise TypeError("The filename must be a string")
        except TypeError as t:
            print(t)
        else:
            # print "i work"
            self.filename = filename
            with open(self.filename, 'r') as myfile:
                self.contents = myfile.read()

    def uniform(self, filename, mode="w",case="upper"):
        '''Write the data to the outfile with uniform case.'''
        try:
            if mode != "w" and mode != "a":
                raise ValueError("Mode needs to be 'w' or 'a'")

            with open(filename, mode) as outfile:
                if case == "upper":
                    outfile.write(self.contents.upper())
                elif case == "lower":
                    outfile.write(self.contents.lower())
                else:
                    raise ValueError("Case needs to be 'lower' or 'upper'")

        except ValueError as v:
            print(v)

    def reverse(self, filename, mode="w", unit="line"):
        '''Write the data to the outfile in reverse order.'''
        try:
            if mode != "w" and mode != "a":
                raise ValueError("Mode needs to be 'w' or 'a'")
            lines = self.contents.split("\n")#Even though contents is 1 string it keeps the new line characters
            if unit == "word":
                #list comprehension
                separate_words = [i.split(" ") for i in lines]#This separates each word in each line
                reverse_words = [i[::-1] for i in separate_words]
                combine_words = [" ".join(i) for i in reverse_words]
                combine_lines = "\n".join(combine_words)
                with open(filename, mode) as outfile:
                    outfile.write(combine_lines)
            elif unit == "line":
                lines = lines[::-1]# REVERSE LINES
                combine_lines = "\n".join(lines)
                with open(filename, mode) as outfile:
                    outfile.write(combine_lines)
            else:
                raise ValueError("Unit needs to be 'word' or 'line'.")
        except ValueError as v:
            print(v)

    def transpose(self, filename, mode="w"):
        '''Write a “transposed” version of the data to the outfile.'''
        try:
            if mode != "w" and mode != "a":
                raise ValueError("Mode needs to be 'w' or 'a'")
            lines = self.contents.split("\n")
            separate_words = [i.split(" ") for i in lines]
            my_list = []
            for i in xrange(len(separate_words[0])):#For i in how many words there are
                columns = []
                for j in xrange(len(separate_words)-1):
                    columns.append(separate_words[j][i])
                my_list.append(columns)

            combine_words = [" ".join(i) for i in my_list]
            combine_lines = "\n".join(combine_words)
            with open(filename, mode) as outfile:
                outfile.write(combine_lines)
        except ValueError as v:
            print(v)

# Magic Methods--------------------------------------------------------
    def __str__(self):
        source = "Source file:\t\t" + self.filename
        total = "\nTotal Characters:\t" + str(len(self.contents))
        alpha = 0
        for i in xrange(len(self.contents)):
            if self.contents[i].isalpha():
                alpha += 1
        alphabetic = "\nAlphabetic Characters:\t" + str(alpha)
        count = 0
        for i in xrange(10):
            count += self.contents.count(str(i))
        numerical = "\nNumerical Characters:\t" + str(count)
        whitespace = "\nWhitespace Characters:\t" + str(self.contents.count("\n")+self.contents.count("\t")+self.contents.count(" "))
        lines = "\nNumeber of Lines:\t" + str(self.contents.count("\n"))

        string = source + total + alphabetic + numerical + whitespace + lines
        return string

