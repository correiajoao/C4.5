import pdb
from c45 import C45

tree = C45("../data/iris/iris.data", "../data/iris/iris.names")
#c1 = C45("../data/diabetes.data", "../data/diabetes.names")

tree.fetchData()
tree.preprocessData()
tree.generateTree()
tree.showTree()
tree.classify(tree.tree)
