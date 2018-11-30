from ParseFile import parsefile
import networkx as nx
import matplotlib.pyplot as plt

class GraphColor:

    def __init__(self, graph_file):
        self.__density = 0
        self.__GraphStruct = nx.Graph()
        self.__maxDeg = 0
        self.__minDeg = 0
        self.meanDeg = 0
        self.__connected = True

        #init the graph
        self.__initialisze_GraphStruct(graph_file)

    #Initialize the graph structure
    def __initialisze_GraphStruct(self, graph_file):

        #Parse a file and return his properties: nb of nodes, nb of edges, the list of nodes, the list of source-destination (= edges)
        # and the weight of each edge
        nb_nodes, nb_edges, nodes, source, destination, weight = parsefile(graph_file)

        for i in range(0, nb_edges):
            print(nodes.index(source[i]), nodes.index(destination[i]))
            self.__GraphStruct.add_edge(nodes.index(source[i]), nodes.index(destination[i]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.Graph)

    def __eq__(self, obj):
        return False

MyGraph = GraphColor("testgraph.txt")