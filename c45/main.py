#!/usr/bin/env python
import pdb
from c45 import C45

c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")

c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()

#instance = {'Pregnancies':0, 'Glucose':167, 'BloodPressure':48, 'SkinThickness':33, 'Insulin':543, 'BMI':32, 'DiabetesPedigreeFunction':0.89, 'Age':30};
#print("Classification outcome: " + c1.classify(instance, c1.tree))
