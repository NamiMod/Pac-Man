# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# edited by : Seyed Nami Modarressi


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import heapq
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
#----------------------------------------------
def depthFirstSearch(problem):
    nodeStack = util.Stack()
    checked = []
    startState = problem.getStartState()
    nodeStack.push(startState)
    nodeStack.push([]) # action of first node = []
    # pop , check , not goal -> push childNodes
    while nodeStack.isEmpty() == 0:
        actions = nodeStack.pop()
        state = nodeStack.pop()
        if problem.isGoalState(state):
            checked.append(state)
            return actions
        else:
            checked.append(state)
            childNodes = problem.getSuccessors(state)
            isChecked = 0
            for child in childNodes:
                for temp in checked:
                    if child[0] == temp:
                        isChecked = 1
                        break
                        # we have checked the node
                if isChecked == 0:
                    nodeStack.push(child[0])
                    childActions = actions + [child[1]]
                    nodeStack.push(childActions)
                else:
                    isChecked = 0
    else:
        return None
#----------------------------------------------
def breadthFirstSearch(problem):
    nodeQueue = util.Queue()
    checked = []
    startState = problem.getStartState()
    item = []
    item = item + [startState]
    item = item + [[]]
    nodeQueue.push(item)
    # pop , check , not goal -> push childNodes
    while nodeQueue.isEmpty() == 0:
        popItem = nodeQueue.pop()
        state = popItem[0]
        actions = popItem[1]
        if problem.isGoalState(state):
            checked.append(state)
            return actions
        else:
            checked.append(state)
            childNodes = problem.getSuccessors(state)
            isChecked = 0
            isInNodes = 0
            for child in childNodes:
                for temp in checked:
                    if child[0] == temp:
                        isChecked = 1
                        break
                        # we have checked the node
                if isChecked == 0 :
                    for temp in nodeQueue.list:
                        if child[0] == temp[0]:
                            isInNodes = 1
                            break
                            # we have the node
                    if isInNodes == 0:
                        childActions = actions + [child[1]]
                        item = []
                        item = item + [child[0]]
                        item = item + [childActions]
                        nodeQueue.push(item)
                    else:
                        isInNodes = 0
                else:
                    isChecked = 0
    else:
        return None
#----------------------------------------------
def getCost(problem , nodePriorityQueue , child):
    cost = -1
    for temp in nodePriorityQueue.heap:
        if child[0] == temp[2][0]:
            cost = problem.getCostOfActions(temp[2][1])
            break
            # we have the node
    return cost

def uniformCostSearch(problem):
    nodePriorityQueue = util.PriorityQueue()
    checked = []
    startState = problem.getStartState()
    item = []
    item = item + [startState]
    item = item + [[]]
    nodePriorityQueue.push(item, 0)
    # pop , check , not goal -> push childNodes with costs
    while nodePriorityQueue.isEmpty() == 0:
        popItem = nodePriorityQueue.pop()
        state = popItem[0]
        actions = popItem[1]
        if problem.isGoalState(state):
            checked.append(state)
            return actions
        else:
            checked.append(state)
            childNodes = problem.getSuccessors(state)
            isChecked = 0
            for child in childNodes:
                for temp in checked:
                    if child[0] == temp:
                        isChecked = 1
                        break
                        # we have checked the node
                if isChecked == 0 :
                    cost = getCost(problem,nodePriorityQueue,child)
                    if cost == -1:
                        item = []
                        item = item + [child[0]]
                        childActions = actions + [child[1]]
                        item = item + [childActions]
                        childPrice = problem.getCostOfActions(childActions)
                        nodePriorityQueue.push(item,childPrice)
                    else:
                        childActions = actions + [child[1]]
                        childNewPrice = problem.getCostOfActions(childActions)
                        if childNewPrice < cost:
                            item = []
                            item = item + [child[0]]
                            item = item + [childActions]
                            nodePriorityQueue.update(item,childNewPrice)
                else:
                    isChecked = 0
    else:
        return None
#----------------------------------------------
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
#----------------------------------------------
def updatePriorityQueue(nodePriorityQueue, item, priority):
    #this code was copied from util.py 
    #I added some minor change to make it a proper function for my a* search
    for index, (p, c, i) in enumerate(nodePriorityQueue.heap):
        if i[0] == item[0] and p > priority:
            del nodePriorityQueue.heap[index]
            nodePriorityQueue.heap.append((priority, c, item))
            heapq.heapify(nodePriorityQueue.heap)
            break
        elif i[0] == item[0] and p <= priority:
            break
    else:
        nodePriorityQueue.push(item, priority)

def aStarSearch(problem, heuristic=nullHeuristic):
    nodePriorityQueue = util.PriorityQueue()
    checked = []
    startState = problem.getStartState()
    item = []
    item = item + [startState]
    item = item + [[]]
    heuristicValue = heuristic(startState, problem)
    nodePriorityQueue.push(item, heuristicValue)
    # pop , check , not goal -> push childNodes with costs
    while nodePriorityQueue.isEmpty() == 0:
        popItem = nodePriorityQueue.pop()
        state = popItem[0]
        actions = popItem[1] 
        if problem.isGoalState(state):
            checked.append(state)
            return actions
        else:
            checked.append(state)
            childNodes = problem.getSuccessors(state)
            isChecked = 0
            for child in childNodes:
                for temp in checked:
                    if child[0] == temp:
                        isChecked = 1
                        break
                        # we have checked the node
                if isChecked == 0 :
                    item = []
                    item = item + [child[0]]
                    childActions = actions + [child[1]]
                    item = item + [childActions]
                    childNewCost = heuristic(child[0], problem) + problem.getCostOfActions(childActions)
                    updatePriorityQueue(nodePriorityQueue,item,childNewCost)
                else:
                    isChecked = 0
    else:
        return None
#----------------------------------------------
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
