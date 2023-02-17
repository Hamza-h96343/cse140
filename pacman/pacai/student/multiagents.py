import math
import random

import pacai.core.distance
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.directions import Directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***


        ghostProximity = 0
        score = 0
        foodPositions = oldFood.asList()
        foodDistances = [pacai.core.distance.manhattan(newPosition, foodPosition) for foodPosition in foodPositions]

        if len(foodDistances) == 0:
            return 0

        foodProximity = min(foodDistances)
        for i in newGhostPos:

            ghostProximity += pacai.core.distance.euclidean(newPosition, i)
        if(action == 'Stop'):
            score -= 500

        if(foodProximity == 0):
            evalScore = ghostProximity  + score
        else:
            evalScore = (ghostProximity / (foodProximity * 1000)) + score

        return successorGameState.getScore() + evalScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")
        for action in legalMoves:
            if (action == "Stop"):
                continue
            temp = self.minValue(gameState.generateSuccessor(0, action), 0, 1)
            if temp > value:
                value = temp
                move = action
        return move

    def maxValue(self, state, depth):
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)
        value = float("-inf")
        legalMoves = state.getLegalActions()
        for action in legalMoves:
            if (action == "Stop"):
                continue
            value = max(value, self.minValue(state.generateSuccessor(0, action), depth, 1))
        return value

    def minValue(self, state, depth, agentIndex):
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)
        value = float("inf")
        legalMoves = state.getLegalActions(agentIndex)
        if agentIndex == state.getNumAgents() - 1:
            for action in legalMoves:
                if (action == "Stop"):
                    continue
                value = min(value, self.maxValue(state.generateSuccessor(agentIndex, action), depth + 1))
        else:
            for action in legalMoves:
                if (action == "Stop"):
                    continue
                value = min(value,self.minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
        return value
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        move = Directions.STOP
        value = float("-inf")
        alpha = float("inf")
        beta = float("-inf")
        for action in legalMoves:
            if (action == "Stop"):
                continue
            temp = self.minValue(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if temp > value:
                value = temp
                move = action
        return move

    def maxValue(self, state, depth, alpha, beta):
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)
        value = float("-inf")
        legalMoves = state.getLegalActions()
        for action in legalMoves:
            if (action == "Stop"):
                continue
            value = max(value, self.minValue(state.generateSuccessor(0, action), depth, 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def minValue(self, state, depth, agentIndex, aplha, beta):
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)
        value = float("inf")
        legalMoves = state.getLegalActions(agentIndex)
        if agentIndex == state.getNumAgents() - 1:
            for action in legalMoves:
                if (action == "Stop"):
                    continue
                value = min(value, self.maxValue(state.generateSuccessor(agentIndex, action), depth + 1, aplha, beta))
                if value < aplha:
                    return value
                beta = min(beta, value)
        else:
            for action in legalMoves:
                if (action == "Stop"):
                    continue
                value = min(value,self.minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, aplha, beta))
                if value < aplha:
                    return value
                beta = min(beta, value)

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
