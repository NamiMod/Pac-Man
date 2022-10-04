# multiAgents.py
# --------------
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
# completed by : Seyed Nami Modarressi (@SNamiMod)


from typing import final
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        minValue = 999999 # this is value is needed in comparing values

        "*** YOUR CODE HERE ***"
        minDistanceToFood = minValue
        
        for ghost in newGhostStates:
            if ghost.scaredTimer >= 1 :
                continue
            if manhattanDistance(newPos,ghost.getPosition()) <= 1:
                return -minValue
        
        for food in newFood.asList():
            temp = manhattanDistance(newPos, food)
            if temp < minDistanceToFood:
                minDistanceToFood = temp
        
        value = 1/minDistanceToFood
        return successorGameState.getScore() + value


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(0, 0, gameState)[0]

    def minimax(self, depth, agent, gameState):
        
        change = 0 # for the first time we should update action and score
        minORmax_action = 0
        minORmax_score = 0

        if agent == gameState.getNumAgents():
            depth = depth + 1
            agent = 0 # PacMan number is 0
            
        if depth == self.depth:
            return [None, self.evaluationFunction(gameState)]
            
        if agent == 0: # Find Max
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.minimax(depth, agent + 1, next_state)[1]
                if score > minORmax_score or change == 0:
                    minORmax_action = action
                    minORmax_score = score
                    change = 1

        else: # Find Min
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.minimax(depth, agent + 1, next_state)[1]
                if score < minORmax_score or change == 0:
                    minORmax_action = action
                    minORmax_score = score
                    change = 1
                        
        if change == 1:
            return [minORmax_action, minORmax_score]
        else:
            return [None, self.evaluationFunction(gameState)]
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minValue = 999999 # this is value is needed in comparing values
        return self.AlphaBeta(0, 0, gameState, -minValue, minValue)[0] #-> alpha = -999999 and beta = 999999

    def AlphaBeta(self, depth, agent, gameState , alpha , beta):
        
        change = 0 # for the first time we should update action and score
        AlphaBeta_action = 0
        AlphaBeta_score = 0

        if agent == gameState.getNumAgents():
            depth = depth + 1
            agent = 0 # PacMan number is 0
            
        if depth == self.depth:
            return [None, self.evaluationFunction(gameState)]
            
        if agent == 0: # Find Max
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.AlphaBeta(depth, agent + 1, next_state , alpha , beta)[1]
                if score > AlphaBeta_score or change == 0:
                    AlphaBeta_action = action
                    AlphaBeta_score = score
                    change = 1
                if alpha < score:
                    alpha = score
                if alpha > beta:
                    break

        else: # Find Min
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.AlphaBeta(depth, agent + 1, next_state , alpha, beta)[1]
                if score < AlphaBeta_score or change == 0:
                    AlphaBeta_action = action
                    AlphaBeta_score = score
                    change = 1
                if beta > score:
                    beta = score
                if alpha > beta:
                    break


        if change == 1:
            return [AlphaBeta_action, AlphaBeta_score]
        else:
            return [None, self.evaluationFunction(gameState)]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(0, 0, gameState)[0]
    
    def expectimax(self, depth, agent, gameState):
        
        change = 0 # for the first time we should update action and score
        expectimax_action = 0
        expectimax_score = 0

        if agent == gameState.getNumAgents():
            depth = depth + 1
            agent = 0 # PacMan number is 0
            
        if depth == self.depth:
            return [None, self.evaluationFunction(gameState)]
            
        if agent == 0: # Find Max
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.expectimax(depth, agent + 1, next_state)[1]
                if score > expectimax_score or change == 0:
                    expectimax_action = action
                    expectimax_score = score
                    change = 1

        else: # Find Min - in this case we should add probability (for ghosts)
            len_action = len(gameState.getLegalActions(agent))
            if len_action == 0:
                p = 1
            else:
                p = 1 / len_action
            for action in gameState.getLegalActions(agent):  # compare score to all other scores (scores of legal actions)
                next_state = gameState.generateSuccessor(agent, action)
                score = self.expectimax(depth, agent + 1, next_state)[1]

                if change == 0:
                    change = 1
                    
                expectimax_action = action
                expectimax_score = expectimax_score + p * score
                        
        if change == 1:
            return [expectimax_action, expectimax_score]
        else:
            return [None, self.evaluationFunction(gameState)]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    we should evaluate states instead of actions and for this we use this parameters :
        1 - number of food
        2 - distances(for food)
        3 - distances(for ghosts)
        4 - distances(for scared ghosts)
        5 - number of capsules
    """
    "*** YOUR CODE HERE ***"

    Position = currentGameState.getPacmanPosition() # PacMan
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    final_value = currentGameState.getScore() # Evaluation value with start value

    final_value = final_value + -5 * len(food) # number of food
    final_value = final_value + -10 * len(currentGameState.getCapsules()) # number of capsules

    for f in food:
        d = manhattanDistance(Position,f)
        if d < 2:
            final_value = final_value + -1 * d
        if d > 5:
            final_value = final_value + -0.1 * d
        else:
            final_value = final_value + -0.5 * d

    for g in ghosts:
        d = manhattanDistance(Position,g.getPosition())
        if g.scaredTimer == 0:
            if d < 2:
                final_value = final_value + -15 * d
            else:
                final_value = final_value + -5 * d
        else:
            if d < 2:
                final_value = final_value + 3 * d
            elif d < 5:
                final_value = final_value + 2 * d
            else:
                final_value = final_value + 0.5 * d

    return final_value

# Abbreviation
better = betterEvaluationFunction
