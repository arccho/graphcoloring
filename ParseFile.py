
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

    return int(nb_nodes), int(nb_edges), nodes, source, destination, weight
