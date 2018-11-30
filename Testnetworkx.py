
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 6), (4, 5), (5, 6)])
plt.subplot(111)

nx.draw(G, with_labels=True, font_weight='bold')

G.nodes[1]['color'] = 'red'
plt.show()
#G.add_weighted_edges_from()
a = G.nodes.data()