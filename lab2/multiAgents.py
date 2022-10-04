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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        foodList = currentGameState.getFood().asList()
        newfoodList = newFood.asList()

        score = 0

        if len(newfoodList) == 0:
            return 1_000_000


        score -= len(newfoodList) * 10
        if len(newfoodList) < len(foodList):
            score += 100

        closestGhost = min([manhattanDistance(currentGameState.getPacmanPosition(), ghostCoord.getPosition())
                            for ghostCoord in currentGameState.getGhostStates()])
        newClosestGhost = min([manhattanDistance(newPos, ghostCoord.getPosition())
                               for ghostCoord in newGhostStates])

        if newClosestGhost < closestGhost:
            score -= 1
            if newClosestGhost <= 2:
                score -= 150

        closestFood = min([manhattanDistance(currentGameState.getPacmanPosition(), foodPos)
                           for foodPos in foodList])
        newClosestFood = min([manhattanDistance(newPos, foodPos) for foodPos in newfoodList])

        if newClosestFood < closestFood:
            score += 10

        if currentGameState.getPacmanPosition() == newPos:
            score -= 10



        return score










def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        n = gameState.getNumAgents()

        def miniMax(state: GameState, agentIndex, depth):
            properAction = None
            actions = state.getLegalActions(agentIndex)
            maxVal = - 10_000_000
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = getMin(successor, agentIndex + 1, depth)
                if value > maxVal:
                    maxVal = value
                    properAction = action
            return properAction


        def getMax(state: GameState, depth):
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            maxVal = - 10_000_000

            actions = state.getLegalActions(0)
            for action in actions:
                successor = state.generateSuccessor(0, action)
                maxVal = max(maxVal, getMin(successor, 1, depth))

            return maxVal



        def getMin(state: GameState, agentIndex, depth):
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            minVal = 10_000_000

            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex < n - 1:
                    minVal = min(minVal, getMin(successor, agentIndex + 1, depth))
                else:
                    minVal = min(minVal, getMax(successor, depth + 1))
            return minVal

        return miniMax(gameState, self.index, 0)





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        n = gameState.getNumAgents()

        def miniMax(state: GameState, agentIndex, depth):
            properAction = None
            actions = state.getLegalActions(agentIndex)
            maxVal = - 10_000_000
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = getMin(successor, agentIndex + 1, depth, maxVal, 10_000_000)
                if value > maxVal:
                    maxVal = value
                    properAction = action
            return properAction

        def getMax(state: GameState, depth, alpha, beta):
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            maxVal = - 10_000_000

            actions = state.getLegalActions(0)
            for action in actions:
                successor = state.generateSuccessor(0, action)
                maxVal = max(maxVal, getMin(successor, 1, depth, alpha, beta))
                if maxVal > beta:
                    break
                alpha = max(alpha, maxVal)
            return maxVal

        def getMin(state: GameState, agentIndex, depth, alpha, beta):
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            minVal = 10_000_000

            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex < n - 1:
                    minVal = min(minVal, getMin(successor, agentIndex + 1, depth, alpha, beta))
                else:
                    minVal = min(minVal, getMax(successor, depth + 1, alpha, beta))
                if minVal < alpha:
                    break
                beta = min(beta, minVal)

            return minVal

        return miniMax(gameState, self.index, 0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        n = gameState.getNumAgents()

        def miniMax(state: GameState, agentIndex, depth):
            properAction = None
            actions = state.getLegalActions(agentIndex)
            maxVal = - 10_000_000
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = getExpected(successor, agentIndex + 1, depth)
                if value > maxVal:
                    maxVal = value
                    properAction = action
            return properAction

        def getExpected(state: GameState, agentIndex, depth):
            if (depth == self.depth) or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if agentIndex == 0:
                expected = - 10_000_000
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    expected = max(expected, getExpected(successor, 1, depth))
                return expected
            else:
                expected = 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)

                    if agentIndex < n - 1:
                        expected += getExpected(successor, agentIndex + 1, depth)
                    else:
                        expected += getExpected(successor, 0, depth + 1)
                expected /= len(actions)
                return expected

        return miniMax(gameState, self.index, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
