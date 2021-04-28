import os


file_prefix = 'output/bialaczka_'

def get_lines_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as myfile:
        return myfile.readlines()


def read_from_file_to_dict(filename):
    return dict(enumerate(read_from_file_to_list(filename), start=1))


def read_from_file_to_list(filename):
    return [line.rstrip('\n') for line in get_lines_from_file(filename)]


def read_parameters_from_file():
    global classes, symptoms
    classes = read_from_file_to_dict('jednostki.txt')
    symptoms = read_from_file_to_list('cechy.txt')



def divide_data_to_files():
    block_counter = 0
    if(not os.path.isdir('output')):
        os.mkdir('output')
    with open('bialaczka.csv', 'r') as myfile:
        block_lines = []
        for line in myfile.readlines():
            index = line.find(';')
            if(index > 0):  # new block
                if(block_counter > 0):  # to avoid empty first block
                    with open(f"{file_prefix}{block_counter}.txt", 'w') as resultfile:
                        resultfile.writelines(block_lines)
                    block_lines = []
                block_counter += 1
            line = line[index + 1:]
            index = line.find(';')
            line = line[index + 1:]
            line = line.replace(';', ' ')
            block_lines.append(line)
        with open(f"{file_prefix}{block_counter}.txt", 'w') as resultfile:  # to get last block
            resultfile.writelines(block_lines)
    global classes_size
    classes_size = block_counter



def parse():
    read_parameters_from_file()
    divide_data_to_files()


if(__name__ == '__main__'):
    parse()