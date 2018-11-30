from ParseFile import parsefile
import networkx as nx
import matplotlib.pyplot as plt

class GraphColor:

    def __init__(self, graph_file):
        self.__graphStruct = nx.Graph()
        self.__connected = False
        self.__listDeg = list()
        self.__minDeg = 0
        self.__maxDeg = 0
        self.__meanDeg = 0.0
        self.__density = 0.0

        #init the graph
        self.__initialisze_GraphStruct(graph_file)

    #Initialize the graph structure
    def __initialisze_GraphStruct(self, graph_file):

        #Parse a file and return his properties: nb of nodes, nb of edges, the list of nodes, the list of source-destination (= edges)
        # and the weight of each edge
        nb_nodes, nb_edges, nodes, source, destination, weight = parsefile(graph_file)

        #fill the graph
        for i in range(0, nb_edges):
            #print(nodes.index(source[i]), nodes.index(destination[i]))
            self.__graphStruct.add_edge(nodes.index(source[i]), nodes.index(destination[i]))

        # init the attributs of class
        for num_node in range(0, nb_nodes):
            #print(len(list(self.__GraphStruct.neighbors(num_node))))
            self.__listDeg.append(len(list(self.__graphStruct.neighbors(num_node))))

        self.__minDeg = min(self.__listDeg)
        self.__maxDeg = max(self.__listDeg)
        self.__meanDeg = float(sum(self.__listDeg)) / len(self.__listDeg)
        #print(self.__minDeg, self.__maxDeg, self.__meanDeg)
        self.__density = nb_edges / float((nb_nodes * (nb_nodes - 1) / 2))
        if self.__minDeg > 0:
            self.__connected = True


    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.GraphColor)

    def __eq__(self, obj):
        return False

MyGraph = GraphColor("testgraph.txt")

# TODO: implementer methode MCMC
