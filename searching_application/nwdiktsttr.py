import heapq

def dijkstra(graph, start, end):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}

    while queue:
        (current_distance, current_node) = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    return distances, previous_nodes

# Modify the graph dynamically
def simulate_failure(graph, node1, node2):
    if node2 in graph[node1]:
        del graph[node1][node2]  # Simulate a failure by removing an edge
    if node1 in graph[node2]:
        del graph[node2][node1]

# Example graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start = 'A'
end = 'D'
print(dijkstra(graph, start, end))

# Simulate a link failure between 'A' and 'B'
simulate_failure(graph, 'A', 'B')
print(dijkstra(graph, start, end))  # Re-run after failure
