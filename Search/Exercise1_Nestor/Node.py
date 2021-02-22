'''
@author: Devangini Patel
Adapted for a simple BFS algorythm
'''

from Relationships import graph

class Node:
    '''
    This class represents a node in the search tree
    '''
    def __init__(self, name, parentNode):
        """
        Constructor
        """
        self.name = name
        self.depth = 0
        self.children = []
        #self.parent = None
        self.setParent(parentNode)
        self.fringe = True
        
        
    def setParent(self, parentNode):
        """
        This method adds a node under another node
        """
        if parentNode != None:
            parentNode.children.append(self)
            self.parent = parentNode
            self.depth = parentNode.depth + 1
        else:
            self.parent = None
        
    def printPath(self):
        """
        This method prints the path from initial state to goal state
        """
        if self.parent != None:
            self.parent.printPath()
        print ("-> ", self.name)
        
        
    