from ParseFile import parsefile
import networkx as nx
import matplotlib.pyplot as plt

class GraphColor:

    def __init__(self, graph_file):
        self.graphStruct = nx.Graph()
        self.connected = False
        self.listDeg = list()
        self.minDeg = 0
        self.maxDeg = 0
        self.meanDeg = 0.0
        self.density = 0.0

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
            self.graphStruct.add_edge(nodes.index(source[i]), nodes.index(destination[i]))

        # init the attributs of class
        for num_node in range(0, nb_nodes):
            try :
                self.listDeg.append(len(list(self.graphStruct.neighbors(num_node))))
            except :
                self.listDeg.append(0)
            #print(self.__listDeg[i])


        self.minDeg = min(self.listDeg)
        self.maxDeg = max(self.listDeg)
        self.meanDeg = float(sum(self.listDeg)) / len(self.listDeg)
        self.density = nb_edges*2 / float((nb_nodes * (nb_nodes - 1) / 2))
        if self.minDeg > 0:
            self.connected = True
        print(self.minDeg, self.maxDeg, self.meanDeg, self.density, self.connected)


    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.GraphColor)

    def __eq__(self, obj):
        return False
