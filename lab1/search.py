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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import searchAgents
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

    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    stack.push((problem.getStartState(), "", 0))
    visited = set()
    prev = dict()
    goal = None
    while not stack.isEmpty():
        current = stack.pop()
        if current[0] in visited:
            continue
        visited.add(current[0])
        if problem.isGoalState(current[0]):
            goal = current
            break
        successors = problem.getSuccessors(current[0])
        for s in successors:
            if s[0] not in visited:
                prev[s[0]] = (current[0], s[1])
                stack.push(s)

    path = []
    if goal is not None:
        current = goal
        while current[0] != problem.getStartState():
            path.append(prev[current[0]][1])
            current = prev[current[0]]
        path.reverse()

    return path


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    queue.push((problem.getStartState(), "", 0))
    visited = set()
    prev = dict()
    goal = None
    visited.add(problem.getStartState())
    while not queue.isEmpty():
        current = queue.pop()

        if problem.isGoalState(current[0]):
            goal = current
            break

        successors = problem.getSuccessors(current[0])
        for s in successors:
            if s[0] not in visited:
                prev[s[0]] = (current[0], s[1])
                queue.push(s)
                visited.add(s[0])

    path = []
    if goal is not None:
        current = goal
        while current[0] != problem.getStartState():
            path.append(prev[current[0]][1])
            current = prev[current[0]]
        path.reverse()

    return path


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    pQueue = util.PriorityQueue()

    used = set()
    g = dict()
    f = dict()
    prev = dict()
    goal = None
    g[problem.getStartState()] = 0
    f[problem.getStartState()] = heuristic(problem.getStartState(), problem)
    pQueue.push((problem.getStartState(), "", 0), f[problem.getStartState()])
    while not pQueue.isEmpty():
        current = pQueue.pop()

        if current[0] in used:
            continue

        if problem.isGoalState(current[0]):
            goal = current
            break

        successors = problem.getSuccessors(current[0])

        for s in successors:
            if g[current[0]] + s[2] < g.get((s[0]), (g[current[0]] + s[2]) + 1):  # if the key is absent, we set it
                g[s[0]] = g[current[0]] + s[2]                                    # to new dist+1 so we can update it
                f[s[0]] = g[s[0]] + heuristic(s[0], problem)
                pQueue.push(s, f[s[0]])
                prev[s[0]] = (current[0], s[1])
        used.add(current[0])

    path = []
    if goal is not None:
        current = goal
        while current[0] != problem.getStartState():
            path.append(prev[current[0]][1])
            current = prev[current[0]]
        path.reverse()

    return path


def greedySearch(problem: SearchProblem, heuristic=nullHeuristic):
    pQueue = util.PriorityQueue()

    used = set()
    used.add(problem.getStartState())
    prev = dict()
    goal = None
    pQueue.push((problem.getStartState(), "", 0), heuristic(problem.getStartState(), problem))
    while not pQueue.isEmpty():
        current = pQueue.pop()

        if problem.isGoalState(current[0]):
            goal = current
            break

        successors = problem.getSuccessors(current[0])
        for s in successors:
            if not s[0] in used:
                used.add(s[0])
                pQueue.push(s, heuristic(s[0], problem))
                prev[s[0]] = (current[0], s[1])


    path = []
    if goal is not None:
        current = goal
        while current[0] != problem.getStartState():
            path.append(prev[current[0]][1])
            current = prev[current[0]]
        path.reverse()

    return path




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch