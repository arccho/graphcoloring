
def parser_parameters(parameters_file):
    parameters = [None] * 5
    file = open("Parameters/" + parameters_file, "rt")
    content = file.read()
    if content is not False:
        line_content = content.split("\n")
        for line in line_content:
            if len(line) > 0:
                element_line = line.split('=')
                if element_line[0].lower() == "epsilon":
                    parameters[0] = element_line[1]
                if element_line[0].lower() == "lambda":
                    parameters[1] = element_line[1]
                if element_line[0].lower() == "ratiofreezed":
                    parameters[2] = element_line[1]
                if element_line[0].lower() == "maxrip":
                    parameters[3] = element_line[1]
                if element_line[0].lower() == "numthreads":
                    parameters[4] = element_line[1]
    return parameters