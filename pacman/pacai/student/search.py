"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util import stack
from pacai.util import queue
from pacai.util import priorityQueue

def helper(datastructure, problem):
    start_state = problem.startingState()

    fringe = datastructure

    fringe.push((start_state, []))

    visited = []

    while not fringe.isEmpty():
        node, direction = fringe.pop()

        if problem.isGoal(node):

            return direction

        for s, a, c in problem.successorStates(node):
            if s not in visited:
                fringe.push((s, direction + [a]))
                visited.append(s)
        visited.append(node)

    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```

    ```
    """
    # *** Your Code Here ***
    return helper(stack.Stack(), problem)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    return helper(queue.Queue(), problem)

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    start_state = problem.startingState()

    fringe = priorityQueue.PriorityQueue()

    fringe.push((start_state, [], 0), 0)

    visited = dict()

    while not fringe.isEmpty():
        node, direction, cost = fringe.pop()

        visited[node] = cost

        if problem.isGoal(node):
            return direction

        for s, a, step_c in problem.successorStates(node):
            if (s not in visited) or (s in visited and visited[s] > cost + step_c):
                visited[s] = cost + step_c

                fringe.push((s, direction + [a], cost + step_c), cost + step_c)

    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    start_state = problem.startingState()

    fringe = priorityQueue.PriorityQueue()

    fringe.push((start_state, [], 0), heuristic(start_state, problem))

    visited = dict()

    while not fringe.isEmpty():
        node, direction, cost = fringe.pop()

        visited[node] = cost

        if problem.isGoal(node):
            return direction

        for s, a, step_c in problem.successorStates(node):
            t_cost = cost + step_c + (heuristic(s, problem))
            if (s not in visited) or (s in visited and visited[s] > t_cost):
                visited[s] = cost + step_c

                fringe.push((s, direction + [a], cost + step_c),
                            cost + step_c + (heuristic(s, problem)))

    return []
