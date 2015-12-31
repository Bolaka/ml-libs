# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:37:42 2015

@author: abzooba
"""

def getInputs(prompt, options):
    """Get comma separated inputs (multiple) from user
    """
    while True:
        value = str(raw_input(prompt))
        strippedValues = []
        if ',' in value:
            values = value.split(',')
            counter = 0
            for val in values:
                stripped = val.strip()
                if len(options) > 0 and stripped not in options:
                    print "Sorry, " + stripped + " is not in: " + ', '.join(options)
                else:
                    counter += 1
                    strippedValues.append(stripped)
            if counter == len(values):
                break
        else:
            continue
            
    return strippedValues

def getInput(prompt, options):
    """Get input (single) from user
    """
    while True:
        value = str(raw_input(prompt))
        if len(options) > 0 and value not in options:
            print "Sorry, " + value + " is not in: " + ', '.join(options)
            continue
        else:
            break
            
    return value 