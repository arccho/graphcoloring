import random
import numpy as np
from collections import Counter
from sys import stdout
import time as tm


def node_generator(number_id, size=12):

    alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    list_nodes = []
    for x in range(number_id):
        list_nodes.append(''.join(random.choice(alphanum) for _ in range(size)))
    list_nodes.sort()
    print str(number_id) + " nodes have been generated with sucessfull"
    return list_nodes


def create_edges(list_nodes, prob_edge):

    percent = 0
    counter = 0
    max_edges = 0
    nb_nodes = len(list_nodes)
    for i in range(1, nb_nodes):
        max_edges += nb_nodes - i
    stdout.write("0% generated")
    list_edge = list()
    for i in range(nb_nodes-1):
        for j in range(i+1, nb_nodes):
            if random.random() <= prob_edge:
                list_edge.append((list_nodes[i], list_nodes[j], str(random.random())))
        counter += nb_nodes - i - 1
        current_percent = int((float(counter) / float(max_edges)) * 100)
        if current_percent != percent:
            percent = current_percent
            stdout.write("\r%d%% generated" % percent)
            stdout.flush()
    stdout.write("\n")
    return list_edge


#slower than create_edgesv1
def create_edgesv2(list_nodes, prob_edge):

    percent = 0
    counter = 0
    max_edges = 0
    nb_nodes = len(list_nodes)
    for i in range(1, nb_nodes):
        max_edges += nb_nodes - i
    list_edge = list()
    stdout.write("0% generated")
    for i in range(nb_nodes-1):
        array_rand = np.random.choice([True, False], size=(nb_nodes-i), p=[prob_edge, 1 - prob_edge])
        index = 0
        for j in range(i+1, nb_nodes):
            if array_rand[index]:
                list_edge.append((list_nodes[i], list_nodes[j], str(random.random())))
            index += 1
        counter += nb_nodes - i - 1
        current_percent = int((float(counter) / float(max_edges)) * 100)
        if current_percent != percent:
            percent = current_percent
            stdout.write("\r%d%% generated" % percent)
            stdout.flush()
    stdout.write("\n")
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
                return False
    return True


def writing_file_graph(name_file, list_node, list_edge):
    print "writing graph in file " + name_file
    file = open(name_file, "wt")

    file.write(str(len(list_node)) + "\t" + str(len(list_edge)) + "\n")
    for edge in list_edge:
        file.write(edge[0] + "\t" + edge[1] + "\t" + edge[2] + "\n")
    file.close()
    print "finished"


def create_graph(name_file, nb_nodes, prob_edge):
    liste_id = node_generator(nb_nodes)
    list_edge = create_edges(liste_id, prob_edge)
    writing_file_graph("Graph/" + name_file + ".txt", liste_id, list_edge)

##################################
###########     EXEC    ##########
##################################

tStart = tm.time()

create_graph("100k", 100000, 0.005)

print('Time: %.1f s' %(tm.time() - tStart))