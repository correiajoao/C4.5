import pdb
from c45 import C45

c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
#c1 = C45("../data/diabetes.data", "../data/diabetes.names")

c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()
c1.classify(c1.tree)


