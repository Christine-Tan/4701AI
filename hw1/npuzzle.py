"""
COMS W4701 Artificial Intelligence - Programming Homework 1

In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now). 
See Courseworks for detailed instructions.

@author: Xinyue Tan (xt2215)
"""

import time


def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped. 
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]

    new_state = []
    for row in range(len(state)):
        new_row = []
        for column in range(len(state[row])):
            if row == i1 and column == j1:
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else:
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)


def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions. 
    The result should be a list containing (Action, state) tuples. 
    For example [("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7))),
                 ("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7)))]
    """
    left = "Left";
    right = "Right";
    up = "Up";
    down = "Down";

    # Hint: Find the "hole" first, then generate each possible
    # successor state by calling the swap_cells method.
    # Exclude actions that are not applicable.

    child_states = []
    l_state = len(state);
    flag = False;
    for i in range(l_state):
        for j in range(l_state):
            if state[i][j] == 0:
                flag = True;
                # left
                if j + 1 < l_state:
                    child_states.append((left, swap_cells(state, i, j, i, j + 1)));
                # right
                if j > 0:
                    child_states.append((right, swap_cells(state, i, j, i, j - 1)));
                # up
                if i + 1 < l_state:
                    child_states.append((up, swap_cells(state, i, j, i + 1, j)));
                # down
                if i > 0:
                    child_states.append((down, swap_cells(state, i, j, i - 1, j)));
                break;
        if flag:
            break;
    return child_states;


def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise. 
    """
    # for a given place in n-puzzle
    # Goal state is: state[i][j] = sqrt(n+1)*i+j
    l_state = len(state);
    for i in range(l_state):
        t = state[i];
        if l_state != len(t):
            print("Invalid n-puzzle!");
            return False;
        else:
            for j in range(len(t)):
                if t[j] != (l_state * i + j):
                    return False;
    return True;


def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a queue in BFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0
    max_fringe = 0;

    fringe = []
    closed = set()
    parents = {}

    # initial the fringe
    origin_state = state;
    fringe.append(state);

    while len(fringe) > 0:
        if len(fringe) > max_fringe:
            max_fringe = len(fringe);

        now_state = fringe.pop(0);
        if now_state in closed:
            # already expanded
            continue;

        # put the state into closed set
        closed.add(now_state);
        states_expanded += 1;
        if not goal_test(now_state):
            # keep expanding
            child_states = get_successors(now_state);
            for index in range(len(child_states)):
                action = child_states[index][0];
                child_state = child_states[index][1];
                # put the state on the fringe if it has not been expanded yet
                if child_state not in closed:
                    fringe.append(child_state);
                    # dict from child_state to (parent state, action) when a state is seen for the first time
                    parents[child_state] = (now_state, action);
        else:
            # return solution, states_expanded, max_fringe
            # find its parent states and actions until it is the original state.
            solution = list();
            while True:
                if now_state == origin_state:
                    break;
                parent_state = parents.get(now_state)[0];
                action = parents.get(now_state)[1];
                solution.append(action);
                now_state = parent_state;

            # reverse solution
            solution = solution[::-1];
            return solution, states_expanded, max_fringe;

    return None, states_expanded, max_fringe  # No solution found


def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a stack in DFS)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}

    # initial the fringe
    origin_state = state;
    fringe.append(state);

    while len(fringe) > 0:
        if len(fringe) > max_fringe:
            max_fringe = len(fringe);

        now_state = fringe.pop();
        if now_state in closed:
            # already expanded
            continue;

        # put the state into closed set
        closed.add(now_state);
        states_expanded += 1;
        if not goal_test(now_state):
            # keep expanding
            child_states = get_successors(now_state);
            for index in range(len(child_states)):
                action = child_states[index][0];
                child_state = child_states[index][1];
                # put the state on the fringe if it has not been expanded yet
                if child_state not in closed:
                    fringe.append(child_state);
                    # dict from child_state to (parent state, action) when a state is seen for the first time
                    parents[child_state] = (now_state, action);
        else:
            # return solution, states_expanded, max_fringe
            # find its parent states and actions until it is the original state.
            solution = [];
            while True:
                if now_state == origin_state:
                    break;
                parent_state = parents.get(now_state)[0];
                action = parents.get(now_state)[1];
                solution.append(action);
                now_state = parent_state;

            # reverse solution
            solution = solution[::-1];
            return solution, states_expanded, max_fringe;

    #  return solution, states_expanded, max_fringe
    return None, states_expanded, max_fringe  # No solution found


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """

    # YOUR CODE HERE
    number = 0;
    l_state = len(state);
    for i in range(l_state):
        t = state[i];
        for j in range(len(t)):
            if t[j] != (l_state * i + j):
                number += 1;
    return number;


def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the Manhattan distance between the current
    position and the goal position. Then return the sum of all distances.
    """
    sum = 0;
    l_state = len(state);
    for i in range(l_state):
        t = state[i];
        for j in range(len(t)):
            if t[j] != (l_state * i + j):
                # compute the manhattan distance
                g_i = t[j] / l_state;
                g_j = t[j] % l_state;
                sum += abs(g_i - i) + abs(g_j - j);
    return sum  # replace this


def best_first(state, heuristic):
    """
    Best first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}

    # initial the fringe
    origin_state = state;
    heappush(fringe, (heuristic(state), state));

    while len(fringe) > 0:
        if len(fringe) > max_fringe:
            max_fringe = len(fringe);

        (cost, now_state) = heappop(fringe);
        if now_state in closed:
            # already expanded
            continue;

        # put the state into closed set
        closed.add(now_state);
        states_expanded += 1;
        if not goal_test(now_state):
            # keep expanding
            child_states = get_successors(now_state);
            for index in range(len(child_states)):
                action = child_states[index][0];
                child_state = child_states[index][1];
                # put the state on the fringe if it has not been expanded yet
                if child_state not in closed:
                    heappush(fringe, (heuristic(child_state), child_state));
                    # dict from child_state to (parent state, action) when a state is seen for the first time
                    parents[child_state] = (now_state, action);
        else:
            # return solution, states_expanded, max_fringe
            # find its parent states and actions until it is the original state.
            solution = list();
            while True:
                if now_state == origin_state:
                    break;
                parent_state = parents.get(now_state)[0];
                action = parents.get(now_state)[1];
                solution.append(action);
                now_state = parent_state;

            # reverse solution
            solution = solution[::-1];
            return solution, states_expanded, max_fringe;

    return None, states_expanded, max_fringe  # No solution found


def astar(state, heuristic):
    """
    A-star search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the fringe.
    You may want to keep track of three mutable data structures:
    - The fringe of nodes to expand (operating as a priority queue in greedy search)
    - A set of closed nodes already expanded
    - A mapping (dictionary) from a given node to its parent and associated action
    """
    # You may want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    states_expanded = 0
    max_fringe = 0

    fringe = []
    closed = set()
    parents = {}
    costs = {}

    # initial the fringe
    origin_state = state;
    costs[state] = 0;
    heappush(fringe, (heuristic(state) + costs[state], state));

    while len(fringe) > 0:
        if len(fringe) > max_fringe:
            max_fringe = len(fringe);

        (f, now_state) = heappop(fringe);
        if now_state in closed:
            # already expanded
            continue;

        # put the state into closed set
        closed.add(now_state);
        states_expanded += 1;
        if not goal_test(now_state):
            # keep expanding
            child_states = get_successors(now_state);
            for index in range(len(child_states)):
                action = child_states[index][0];
                child_state = child_states[index][1];
                # put the state on the fringe if it has not been expanded yet
                if child_state not in closed:
                    costs[child_state] = costs.get(now_state) + 1;
                    heappush(fringe, (heuristic(child_state) + costs[child_state], child_state));
                    # dict from child_state to (parent state, action) when a state is seen for the first time
                    parents[child_state] = (now_state, action);

        else:
            # return solution, states_expanded, max_fringe
            # find its parent states and actions until it is the original state.
            solution = list();
            while True:
                if now_state == origin_state:
                    break;
                parent_state = parents.get(now_state)[0];
                action = parents.get(now_state)[1];
                solution.append(action);
                now_state = parent_state;

            # reverse solution
            solution = solution[::-1];
            return solution, states_expanded, max_fringe;
    return None, states_expanded, max_fringe  # No solution found


def print_result(solution, states_expanded, max_fringe):
    """
    Helper function to format test output. 
    """
    if solution is None:
        print("No solution found.")
    else:
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max fringe size: {}.".format(max_fringe))


if __name__ == "__main__":
    # Easy test case
    # test_state = ((1, 4, 2),
    #               (0, 5, 8),
    #               (3, 6, 7))

    # More difficult test case
    test_state = ((7, 2, 4),
                 (5, 0, 6),
                 (8, 3, 1))

    print(state_to_string(test_state))
    print()

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_fringe = bfs(test_state)  #
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end - start))

    print()
    print("====DFS====")
    start = time.time()
    solution, states_expanded, max_fringe = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end - start))

    print()
    print("====A* (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end - start))

    print()
    print("====A* (Total Manhattan Distance Heuristic)====")
    start = time.time()
    solution, states_expanded, max_fringe = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_fringe)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end - start))
