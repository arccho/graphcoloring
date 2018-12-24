from ParseFile import parsefile
import networkx as nx
import pycuda.driver as cuda
from pycuda.characterize import sizeof
from pycuda import gpuarray
import matplotlib.pyplot as plt
import numpy as np

class GraphColor:

    def __init__(self, graph_file):
        self.graphStruct = nx.Graph()
        self.connected = False
        self.listDeg = list()
        self.minDeg = 0
        self.maxDeg = 0
        self.meanDeg = 0.0
        self.density = 0.0

        self.node_source = list()
        self.node_destination = list()
        self.edges = list()


        #init the graph
        self.__initialize_GraphStruct(graph_file)
        self.__initialize_GPU()

    #Initialize the graph structure
    def __initialize_GraphStruct(self, graph_file):

        #Parse a file and return his properties: nb of nodes, nb of edges, the list of nodes, the list of source-destination (= edges)
        # and the weight of each edge
        nb_nodes, nb_edges, nodes, source, destination, weight = parsefile(graph_file)

        #fill the graph
        for i in range(0, nb_edges):
            self.node_source.append(nodes.index(source[i]))
            self.node_destination.append(nodes.index(destination[i]))
            self.graphStruct.add_edge(self.node_source[i], self.node_destination[i])

        self.edges = list()
        self.listDeg = [0] * nb_nodes
        for num_node in self.node_source:
            self.listDeg[num_node] = len(list(self.graphStruct.neighbors(num_node)))

        for num_node in self.node_destination:
            self.listDeg[num_node] = len(list(self.graphStruct.neighbors(num_node)))


        for index in range(len(self.node_source)):
            if self.node_source[index] < self.node_destination[index]:
                self.edges.append(self.node_source[index])
                self.edges.append(self.node_destination[index])

        self.minDeg = min(self.listDeg)
        self.maxDeg = max(self.listDeg)
        self.meanDeg = float(sum(self.listDeg)) / len(self.listDeg)
        self.density = len(self.edges) / float((nb_nodes * (nb_nodes - 1) / 2))
        if self.minDeg > 0:
            self.connected = True
        print(self.minDeg, self.maxDeg, self.meanDeg, self.density, self.connected)

    def __initialize_GPU(self):
        self.cuda_edges = gpuarray.to_gpu(np.array(self.edges))
        self.cuda_listDeg = gpuarray.to_gpu(np.array(self.listDeg))



    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.GraphColor)

    def __eq__(self, obj):
        return False
