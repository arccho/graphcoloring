from ParseFile import parsefile
from pycuda import gpuarray
import numpy as np
import time as tm

class GraphColor:

    def __init__(self, graph_file):
        self.connected = False
        self.listDeg = list()
        self.cumulDeg = list()
        self.minDeg = 0
        self.maxDeg = 0
        self.meanDeg = 0.0
        self.density = 0.0

        self.node_source = list()
        self.node_destination = list()
        self.edges = list()

        self.nb_nodes = 0
        self.nb_edges = 0


        #init the graph
        self.__initialize_GraphStruct(graph_file)

    #Initialize the graph structure
    def __initialize_GraphStruct(self, graph_file):

        tStart = tm.time()
        ###################################
        ##########   PART CPU    ##########
        ###################################

        #Parse a file and return his properties: nb of nodes, nb of edges, the list of nodes, the list of source-destination (= edges)
        # and the weight of each edge
        print("parsing file ...")
        nb_nodes, nb_edges, dic_nodes, source, destination, weight = parsefile(graph_file)
        print("end parsing")
        print('Time: %.1f s' % (tm.time() - tStart))

        self.nb_nodes = len(dic_nodes)
        self.nb_edges = len(source)

        #fill the graph
        tStart = tm.time()
        print("start filling source and destination")
        for i in range(0, nb_edges):
            self.node_source.append(dic_nodes[source[i]])
            self.node_destination.append(dic_nodes[destination[i]])
        print("end filling source and destination")
        print('Time: %.1f s' % (tm.time() - tStart))

        #list of all neighbors of all nodes
        self.listNeighbors = list()
        self.listDeg = [0] * nb_nodes
        tStart = tm.time()
        print("start list of neighbors")
        for num_node in range(nb_nodes):
            self.listNeighbors.append([])
            for index in range(len(self.node_source)):
                if self.node_source[index] == num_node:
                    self.listNeighbors[num_node].append(self.node_destination[index])
                if self.node_destination[index] == num_node:
                    self.listNeighbors[num_node].append(self.node_source[index])
            #print self.listNeighbors[num_node]
            self.listDeg[num_node] = len(self.listNeighbors[num_node])
        #print self.listDeg
        print("end list of neighbors")
        print('Time: %.1f s' % (tm.time() - tStart))

        #Cumul list of neighbors numbers for all nodes
        tStart = tm.time()
        print("start cumul list of neighbors")
        temp = list(self.listDeg)
        temp.insert(0, 0)
        self.cumulDeg = np.cumsum(temp)
        print("end cumul list of neighors")
        print('Time: %.1f s' % (tm.time() - tStart))
        #print self.cumulDeg

        tStart = tm.time()
        print("start list of edges")
        #Edges in single list
        for i in range(nb_nodes):
            for j in self.listNeighbors[i]:
                if i < j:
                    self.edges.append(i)
                    self.edges.append(j)
        #print self.edges
        print("end list of edges")
        print('Time: %.1f s' % (tm.time() - tStart))

        self.minDeg = min(self.listDeg)
        self.maxDeg = max(self.listDeg)
        self.meanDeg = float(sum(self.listDeg)) / len(self.listDeg)
        self.density = len(self.edges) / float((nb_nodes * (nb_nodes - 1) / 2))
        if self.minDeg > 0:
            self.connected = True
        print (self.nb_nodes, self.nb_edges)
        print(self.minDeg, self.maxDeg, self.meanDeg, self.density, self.connected)

        ###################################
        ##########   PART GPU    ##########
        ###################################

        #list of all edges
        self.cuda_edges = gpuarray.to_gpu(np.array(self.edges))

        self.cuda_listCumulDeg = gpuarray.to_gpu(self.cumulDeg)
        #print self.cuda_listCumulDeg.get()

        #List of all neighbors. We can find all neighbors of each node in this list in using listCumulDeg
        #Example: for node 3:  listCumulDeg[3] return the index of his neighbors in listNeihbors
        #We know the number of neighbors with (listDeg[3]) or (listCumulDeg[3] - listCumulDeg[3-1])
        temp = list()
        for list_neighbors in self.listNeighbors:
            #print list_neighbors
            temp.extend(list_neighbors)
        self.cuda_listNeighbors = gpuarray.to_gpu(np.array(temp))
        #print self.cuda_listNeighbors.get()


    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.GraphColor)

    def __eq__(self, obj):
        return False
