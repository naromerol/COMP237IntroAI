'''
@author: Devangini Patel
'''
from State import State
from Node import Node
import queue
from TreePlot import TreePlot
    

def performAStarSearch():
    """
    This method performs A* search
    """
    
    #create queue
    pqueue = queue.PriorityQueue()
    
    #create root node
    initialState = State()
    root = Node(initialState, None)
    
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)
    
    #Visited nodes
    visited = {}
    
    #add to priority queue
    pqueue.put((root.costFromRoot, root))
    visited[root.state.place] = root
    
    #check if there is something in priority queue to dequeue
    while not pqueue.empty(): 
        
        #dequeue nodes from the priority Queue
        _, currentNode = pqueue.get()
        
        #remove from the fringe
        currentNode.fringe = False
        
        #check if it has goal State
        print ("-- dequeue --", currentNode.state.place)
        
        #check if this is goal state
        if currentNode.state.checkGoalState():
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            
            #show the search tree explored so far
            treeplot = TreePlot()
            treeplot.generateDiagram(root, currentNode)
            break
            
        #get the child nodes 
        childStates = currentNode.state.successorFunction()
        
        #mark parent as visited to avoid reverse paths
        visited[currentNode.state.place] = currentNode
        
        for childState in childStates:
            
            childNode = Node(State(childState), currentNode)
                        
            if(not(childNode.state.place in visited )):
                #add to tree and queue
                print("Adding node to pQueue")
                print((childNode.costFromRoot, childNode.state.place))
                pqueue.put((childNode.costFromRoot, childNode))
            
        #show the search tree explored so far
        treeplot = TreePlot()
        treeplot.generateDiagram(root, currentNode)
        
                
    #print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
performAStarSearch()