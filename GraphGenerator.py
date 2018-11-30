import random
import numpy
from collections import Counter

def node_generator(number_id, size=12):
    alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    list_nodes = []
    for x in range(number_id):
        list_nodes.append(''.join(random.choice(alphanum) for _ in range(size)))
    return list_nodes

def create_edges(list_nodes, prob_edge):
    list_edge = []
    for i in list_nodes:
        for j in list_nodes:
            if numpy.random.choice((True, False), p=[prob_edge, 1-prob_edge]) == True:
                list_edge.append((i, j))
    return list_edge

def check_unique_nodes(list_node):
    counter = Counter(list_node)
    for values in counter.itervalues():
        if values > 1:
            return False
    return True

def check_unique_edge(list_edge):
    for i in range(len(list_edge)):
        for y in range(len(list_edge)):
            if i != y and list_edge[i][0] == list_edge[y][1] and list_edge[i][1] == list_edge[y][0]:
                print(i, y)
                return False
    return True

def writing_file_graph(name_file, list_node, list_edge):
    if not check_unique_nodes(list_node):
        raise Exception("Error, nodes duplicate")
    if not check_unique_edge(list_edge):
        print("Error edges duplicate")

    file = open(name_file, "wt")

    file.write(str(len(list_node)) + "\t" + str(len(list_edge)) + "\n")
    for edge in list_edge:
        file.write(edge[0] + "\t" + edge[1] + "\t" + str(random.uniform(0, 1)) + "\n")
    file.close()


##### EXEC
liste_id = node_generator(10)
list_edge = create_edges(liste_id, 0.2)

writing_file_graph("testgraph.txt", liste_id, list_edge)