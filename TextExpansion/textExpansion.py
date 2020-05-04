import sys
import main

def read_samples(dataset_path):
    input_file = open(dataset_path)
    samples = [l.strip("\n") for l in input_file.readlines()]
    input_file.close()

    return samples

def command_line():

    output = ""

    if len(sys.argv) == 5:

        if sys.argv[1] == "-i":
            dataset_path = sys.argv[2]

            if sys.argv[3] == "-p":
                parameter_path = sys.argv[4]

        elif sys.argv[1] == "-p":
            parameter_path = sys.argv[2]

            if sys.argv[3] == "-i":
                dataset_path = sys.argv[4]

    elif len(sys.argv) == 7:

        if sys.argv[1] == "-i":
            dataset_path = sys.argv[2]

            if sys.argv[3] == "-p":
                parameter_path = sys.argv[4]

                if sys.argv[5] == "-o":
                    output = sys.argv[6]

            elif sys.argv[3] == "-o":
                output = sys.argv[4]

                if sys.argv[5] == "-p":
                    parameter_path = sys.argv[6]


        elif sys.argv[1] == "-p":
            parameter_path = sys.argv[2]

            if sys.argv[3] == "-i":
                dataset_path = sys.argv[4]

                if sys.argv[5] == "-o":
                    output = sys.argv[6]

            elif sys.argv[3] == "-o":
                output = sys.argv[4]

                if sys.argv[5] == "-i":
                    dataset_path = sys.argv[6]

        elif sys.argv[1] == "-o":
            output = sys.argv[2]

            if sys.argv[3] == "-i":
                dataset_path = sys.argv[4]

                if sys.argv[5] == "-p":
                    parameter_path = sys.argv[6]

            elif sys.argv[3] == "-p":
                parameter_path = sys.argv[4]

                if sys.argv[5] == "-i":
                    dataset_path = sys.argv[6]

    else:
        print ("Erro\n.")

    if output == "":
    	output = dataset_path

    samples = read_samples(dataset_path)

    parameters = open(parameter_path).read()

    parameters = parameters.lower()
    parameters = parameters.replace(" ", "")
    parameters = parameters.replace("\t", "")

    original = False
    normalization = False
    use_wordnet = False
    use_wikipedia = False
    concepts = False
    disambiguation = False
    All = False

    if 'all=yes' in parameters:
        All = True

    if 'original=yes' in parameters:
        original = True

    if 'normalization=yes' in parameters:
        normalization = True

    if 'concepts=yes' in parameters:
        concepts = True

    if 'disambiguation=yes' in parameters:
        disambiguation = True

    if 'use_wordnet=yes' in parameters:
        use_wordnet = True

    if 'use_wikipedia=yes' in parameters:
        use_wikipedia = True

    if not use_wordnet and not use_wikipedia:
        use_wordnet = use_wikipedia = True


    parameter_file = open(parameter_path, "r")
    lines = [l.strip("\n") for l in parameter_file.readlines()]

    pos = 0
    for l in lines:
        if l.upper() == "CUSTOM_DICTIONARIES":
            break;
        else:
            pos = pos + 1

    custom_dictionaries = []

    for i in range(pos + 1, len(lines)):
        if len(lines[i]) > 0 and lines[i][0] != '#':
            custom_dictionaries.append(lines[i])

    main.translate(samples, normalization, original, concepts, use_wordnet, use_wikipedia, disambiguation, All, custom_dictionaries, output)



if __name__ == "__main__":
    command_line()
