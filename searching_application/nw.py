graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
    'D': [('B', 5), ('C', 8), ('E', 2)],
    'E': [('C', 10), ('D', 2)]
}
import heapq

def ucs(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, []))
    visited = set()
    
    while queue:
        cost, current_node, path = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        path = path + [current_node]
        
        if current_node == goal:
            return cost, path
        
        for neighbor, neighbor_cost in graph[current_node]:
            if neighbor not in visited:
                heapq.heappush(queue, (cost + neighbor_cost, neighbor, path))
    
    return float('inf'), []

result = ucs(graph, 'A', 'E')
print(result)
def heuristic(node, goal):
    # You can replace this with a real heuristic, such as geographical distance
    return abs(ord(goal) - ord(node))

def a_star(graph, start, goal):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, []))
    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)
        path = path + [current]

        if current == goal:
            return g, path

        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                g_new = g + cost
                f_new = g_new + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_new, g_new, neighbor, path))

    return float('inf'), []

result = a_star(graph, 'A', 'E')
print(result)
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Add nodes and edges
G.add_weighted_edges_from([('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1), ('B', 'D', 5), ('C', 'D', 8), ('D', 'E', 2), ('C', 'E', 10)])

# Draw graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
nx.draw_networkx_edge_labels(G, pos, edge_labels={('A', 'B'): 4, ('A', 'C'): 2, ('B', 'C'): 1, ('B', 'D'): 5, ('C', 'D'): 8, ('D', 'E'): 2, ('C', 'E'): 10})
plt.show()
