# -*- coding: utf-8 -*-
"""
COMP237 - Introduction to AI
Search Assignment - Ex1
@author: NestorRomero
"""

from Relationships import graph
from Node import *
import queue

def BFS_Nestor(graph, start_student, end_student ):
    
    #Control variables for input and special conditions
    start_exists = False
    end_exists = False
    path_exists = False
    
    start_exists = start_student in graph.keys()
    end_exists = end_student in graph.keys()
    
    #Initial input validation
    if not(start_exists):
        print("ERROR: Initial student does not exist in our records")
    
    if not(end_exists):
        print("ERROR: End student does not exist in our records")    

    #if any parameter is missing the program exits
    if not(start_exists and end_exists):
        return
    
    #simple case of start student is the same as end_student
    if start_student == end_student:
        print("NOTICE: Start and End student are the same")
        return
        
    print("Relationships for (start)\t", start_student, " >>\t",graph[start_student])
    print("Relationships for (end)  \t", end_student, " >>\t",graph[end_student])
    
    #build the search objects using the given parameters
    path = []
    path.append(start_student)
    
    #Stores visited node names and their depth level depeding on the search tree created
    visited = {}
        
    StartNode = Node(start_student, None)
    fringe =  queue.Queue()
    fringe.put(StartNode)
      
    
    while not fringe.empty():
        
        current = fringe.get()
        
        if current.name != end_student :
            
           if not(current.name in visited) : 
               visited[current.name] = current.depth
               for s in graph[current.name]:
                   childNode = Node(s, current)
                   fringe.put(childNode)    
        else:
            print("\n")
            print("end_student found")
            print("PATH")
            current.printPath()
            print("\n")
            break
    
    print("Visited Nodes: ")
    print(visited)

BFS_Nestor(graph,"George", "Bob")