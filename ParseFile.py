import numpy as np

def parsefile(name_file):
    source = list()
    destination = list()
    weight = list()
    nb_nodes = 0
    nb_edges = 0

    file = open(name_file, "rt")
    content = file.read()
    if content is not False:

        line_content = content.split("\n")
        nb_nodes = 0
        nb_edges = 0
        try:
            nb_nodes, nb_edges = line_content[0].split('\t')
        except:
            nb_nodes, nb_edges = line_content[0].split(' ')

        del line_content[0]
        for line in line_content:
            if len(line)>0:
                element_line = line.split('\t')
                source.append(element_line[0])
                destination.append(element_line[1])
                weight.append(element_line[2])

    #union of source + destination = list of nodes
    nodes = set(source)
    nodes |= set(destination)
    nodes = list(nodes)
    nodes.sort()

    #dictionary of nodes with the key, the string id of node and value its id number
    dic_nodes = {}
    for index in range(len(nodes)):
        dic_nodes[nodes[index]] = index


    #convert to numpy array => commente car plus lent
    #len_id_node = str(len(source[0]))
    #nodes = np.sort(np.array(nodes, dtype='|S' + len_id_node))
    #source = np.array(source, dtype='|S' + len_id_node)
    #destination = np.array(destination, dtype='|S' + len_id_node)
    #weight = np.array(weight)

    return len(nodes), len(source), dic_nodes, source, destination, weight
