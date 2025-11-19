
import heapq

# Graph
graph = {
    'A': [('B', 1), ('C', 1)],
    'B': [('D', 1), ('E', 1)],
    'C': [('F', 1)],
    'D': [],
    'E': [('G', 1)],
    'F': [('G', 1)],
    'G': []
}

# Heuristic values
h = {
    'A': 3, 'B': 2, 'C': 2, 'D': 5, 'E': 1, 'F': 1, 'G': 0
}

def astar(start, goal):

    # Priority queue: (f, g, state)
    frontier = [(h[start], 0, start)]
    visited = {start: 0}

    while frontier:
        f, g, state = heapq.heappop(frontier)

        if state == goal:
            return state

        for next_state, cost in graph[state]:
            new_g = g + cost
            new_f = new_g + h[next_state]

            if next_state not in visited or new_g < visited[next_state]:
                visited[next_state] = new_g
                heapq.heappush(frontier, (new_f, new_g, next_state))

    return None

# Run it
print("A* result:", astar('A', 'G'))

#BFS
from collections import deque

# Graph with costs
graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 4), ('E', 1)],
    'C': [('F', 1)],
    'D': [],
    'E': [('G', 3)],
    'F': [('G', 10)],
    'G': []
}

def bfs_with_cost(start, goal):
    queue = deque([(start, 0)])     # (state, cost_so_far)
    visited = {start: 0}            # best known cost to each state

    while queue:
        state, cost = queue.popleft()

        if state == goal:
            return state, cost

        for next_state, step_cost in graph[state]:
            new_cost = cost + step_cost

            # Only visit if new or cheaper path found
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost
                queue.append((next_state, new_cost))

    return None, None

# Run it
goal, total_cost = bfs_with_cost('A', 'G')
print("BFS Result:", goal)
print("Total Cost:", total_cost)






#DFS
# Graph with costs: state -> [(next_state, cost)]
graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 4), ('E', 1)],
    'C': [('F', 1)],
    'D': [],
    'E': [('G', 3)],
    'F': [('G', 10)],
    'G': []
}

def dfs_with_cost(start, goal):
    # Stack stores (state, cost_so_far)
    stack = [(start, 0)]
    
    # Track visited states
    visited = set()

    while stack:
        state, cost = stack.pop()

        if state == goal:
            return state, cost

        if state not in visited:
            visited.add(state)

            # Explore neighbors (LIFO → DFS)
            for next_state, step_cost in graph[state]:
                new_cost = cost + step_cost
                stack.append((next_state, new_cost))

    return None, None

# Run DFS
goal, total_cost = dfs_with_cost('A', 'G')
print("DFS Result:", goal)
print("Total Cost:", total_cost)