import sys
import math
import logging
from copy import copy

logging.basicConfig(stream=sys.stdout, format='', level=logging.INFO, datefmt=None)

class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToData,pathToNames):
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None

		self.train = []
		self.test = []

		# === Just for log prupose ===
		self.splitCounter = 0
		#=============================

	def fetchData(self):
		logging.info("FETCHING ATTRIBUTES SETTINGS ...")

		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]

			logging.info("	Classes: {}".format(self.classes))
			logging.info("	Number of classes: {}".format(len(self.classes)))
			
			#add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attrValues[attribute] = values
				self.attributes.append(attribute)
			
		self.numAttributes = len(self.attrValues.keys())
		logging.info("	Number of atributes: {}".format(self.numAttributes))
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)

	def preprocessData(self):
		# === Just for log prupose ===
		AttrContinuous = []
		AttrDiscrete = []
		#=============================

		logging.info("PRE-PROCESSING DATA ...")

		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					AttrContinuous.append(self.attributes[attr_index])
					self.data[index][attr_index] = float(self.data[index][attr_index])
				else:
					AttrDiscrete.append(self.attributes[attr_index])
		

		self.test = self.data[0:int(len(self.data)*0.1)]
		self.train = self.data[int(len(self.data)*0.1):int(len(self.data))]		

		for i in self.train:
			logging.info("	Train: {}".format(i));

		for i in self.test:
			logging.info("	Test: {}".format(i));

		self.formatInstancesToTest()

		logging.info("	Continuous attributes: {}".format(set(AttrContinuous)));
		logging.info("	Discrete attributes: {}".format(set(AttrDiscrete)));
		
		logging.info("	Data size: {}".format(len(self.data)));
		logging.info("	Train size: {}".format(len(self.train)));
		logging.info("	Test size: {}".format(len(self.test)));
		
	def formatInstancesToTest(self):
		atrAux = copy(self.attributes)
		atrAux.append("outcome")
		self.test = [dict(zip(atrAux, values)) for values in self.test]

	def printTree(self):
		logging.info("PRINTING TREE ...")
		self.printNode(self.tree)
		
	def printNode(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent +"("+ str(node.sourceSplit) +") " + node.label + " = " + attributes[index] + " : " + child.label + "  " + str(node.infoGain) )
					else:
						print(indent +"("+ str(node.sourceSplit) +") " + node.label + " = " + attributes[index] + " : " + str(node.infoGain))
						self.printNode(child, indent + "	")
			else:
				#numerical
				leftChild = node.children[0]
				rightChild = node.children[1]
				if leftChild.isLeaf:
					print(indent +"("+ str(node.sourceSplit) +") " + node.label + " <= " + str(node.threshold) + " : " + leftChild.label + "  " + str(node.infoGain))
				else:
					print(indent +"("+ str(node.sourceSplit) +") "+ node.label + " <= " + str(node.threshold)+" : " + str(node.infoGain))
					self.printNode(leftChild, indent + "	")

				if rightChild.isLeaf:
					print(indent +"("+ str(node.sourceSplit) +") " + node.label + " > " + str(node.threshold) + " : " + rightChild.label + "  " + str(node.infoGain))
				else:
					print(indent +"("+ str(node.sourceSplit) +") " + node.label + " > " + str(node.threshold) + " : " + str(node.infoGain))
					self.printNode(rightChild , indent + "	")

	def generateTree(self):
		logging.info("BUILDING TREE ...");
		self.tree = self.recursiveGenerateTree(self.train, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		logging.info("	Attributes: {}".format(curAttributes))
		
		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None, None, -1)

		elif len(curAttributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, majClass, None, None, -1)

		elif self.allSameClass(curData) is not False:
			#return a node with that class
			logging.info("	No split, all data has the same class: {} ".format(curData[-1]))
			return Node(True, self.allSameClass(curData), None, None, -1)
		else:

			(best, best_threshold, splitted, maxEnt, sourceSplit) = self.splitAttribute(curData, curAttributes)
			
			logging.info("")
			logging.info(" ---------------------------------------------------------------------------------------------")
			logging.info("	Split id: {}".format(sourceSplit))
			logging.info("	Best attribute: {}".format(best))
			logging.info("	Best thresoulder: {}".format(best_threshold))
			logging.info("	Max gain: {}".format(maxEnt))
			logging.info(" ---------------------------------------------------------------------------------------------")
			logging.info("")

			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold, maxEnt, sourceSplit)
			node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif (len(self.attrValues[attribute]) == 1) and (self.attrValues[attribute][0] == "continuous"):
			return False
		else:
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		splitId = -1
		maxEnt = -1*float("inf")
		best_attribute = -1
		best_threshold = None
		
		for attribute in curAttributes:

			self.splitCounter += 1;
			logging.info("	Split: {} -- Attribute: {}".format(self.splitCounter, attribute))
			
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row == valuesForAttribute[index]:
							subsets[index].append(row)
							break

				e = self.gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)

						e = self.gain(curData, [less, greater])
						logging.info("			Current info gain: {} -- Threshould: {}".format(e, threshold));

						if e >= maxEnt:
							splitted = [less, greater]
							splitId = self.splitCounter
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold

		return (best_attribute,best_threshold,splitted, maxEnt, splitId)

	def gain(self,unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)

		#logging.info(	"Impurity before: {}".format(impurityBeforeSplit))
		#calculate impurity after split
		weights = [float(len(subset))/float(S) for subset in subsets]
		impurityAfterSplit = 0

		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
	
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def entropy(self, dataSet):
		S = len(dataSet)
		
		if S == 0:
			return 0
		
		num_classes = [0 for i in self.classes]

		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1

		num_classes = [float(x)/float(S) for x in num_classes]
		return sum(-p * math.log(p,2) for p in num_classes if p)

	def classify(self, tree):
		logging.info("CLASSIFICATION RESULT ...");
		for _class in self.classes:
			correct = 0;
			wrong = 0;

			for instance in self.test:
				if instance["outcome"] == _class: 
					
					classification = self.classifyInstance(copy(instance), tree, False) 

					if classification == instance['outcome']:
						logging.info("	Instance: {} -- Classification Result: {}".format(instance, classification))
						correct += 1;
					else:
						logging.info("	(WRONG) Instance: {} -- Classification Result: {}".format(instance, classification))
						wrong += 1;

					self.classifyInstance(copy(instance), tree, True) 

			logging.info("		Correct: {}".format(correct))
			logging.info("		Wrong: {}".format(wrong))
			logging.info("		Accuracy: {}".format(float(correct)/float(wrong+correct)))
			logging.info("		")


	def classifyInstance(self, instance, tree, log):

		if tree.isLeaf:
			return tree.label
		if(tree.threshold is None):
				print("Not implemented")
		else:
			if(instance[tree.label] <= tree.threshold):
				if log:
					logging.info("		{} {} <= {}".format(tree.label, instance[tree.label], tree.threshold))

				del instance[tree.label]
				return self.classifyInstance(instance, tree.children[0], log)
			elif (instance[tree.label] > tree.threshold):
				if log:
					logging.info("		{} {} > {}".format(tree.label, instance[tree.label], tree.threshold))

				del instance[tree.label]
				return self.classifyInstance(instance, tree.children[1], log)

class Node:
	def __init__(self,isLeaf, label, threshold, infoGain, sourceSplit):
		self.label = label
		self.threshold = threshold
		self.infoGain = infoGain
		self.isLeaf = isLeaf
		self.children = []
		
		# === Just for log prupose ===
		self.sourceSplit = sourceSplit
		#=============================