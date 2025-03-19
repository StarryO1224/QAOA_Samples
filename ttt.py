from graph_generator import generate_3_regular_graph


graph = [(0,1),(0,3),(1,2),(2,3),(3,4)]
print(len(graph))

G = generate_3_regular_graph(4)
print(len(G.edges()))