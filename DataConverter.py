'''
Prashant Sridhar (54565839)
Project 2

This python module is used to convert csv or tsv data files into formats usable for the associated networks. Data is
one-hot-encoded when applicable, with inputs on one line and output on the next.
'''

import re

def get_column_values(column_number, data):
    ''' Determine all possible values a column of data holds in order to determine
        a nominal data type. '''

    value_list = {}     # Dictionary to hold all the classes present in the data

    # checks the value contained in column in each line, adding new values to the dict
    for line in data.split('\n'):
        line = line.split(',')
        if len(line) > 1: value_list.update({line[column_number]: 1})

    try:
        # if the column contains only real values the function returns true
        for key in value_list.keys():
            float(key)
        return (True, value_list)
    except ValueError:
        # else returns false and a dictionary of nominal attributes
        return (False, value_list)


def one_hot_encode(dict):
    ''' Creates a dictionary to one-hot-encode nominal attributes'''
    counter = 0
    for key in dict.keys():
        dict[key] = ''
        for i in range(counter): dict[key] += '0,'
        dict[key] += '1,'
        counter += 1
        for i in range(len(dict.keys()) - counter): dict[key] += '0,'
        dict[key] = dict[key][:-1]
    return dict


def process_file(name_in, name_out, nominal_columns, remove_column):
    ''' Takes input file of a csv or tsv format and formats the value into one-hot-encoding values, with the input
        values on one line, and their resulting class value on the next, all comma separated. '''

    fin = open(name_in, 'r')
    fout = open(name_out, 'w')
    data = ''

    column_types = []
    encoder = []

    for line in fin:
        line = re.sub("\s+", ',', line.strip())  # strips all white space, adding commas to separate values
        data += line + '\n'  # adds the converted line to the data string

    # used to remove a column from the data
    if remove_column != -1:
        old_data = data
        data = ''
        for line in old_data.split('\n'):
            line = line.split(',')
            line.pop(remove_column)
            data += ','.join(line) + '\n'

    sample = data.split('\n')
    sample = sample[0].split(',')

    for column in range(len(sample)):
        real, dict = get_column_values(column, data)

        # translate integer encoding to one-hot encoding
        if column in nominal_columns:
            real = False
        column_types.append(real)

        if not real :
            dict = one_hot_encode(dict)
            encoder.append(dict.copy())
        else:
            encoder.append({}.copy())


    for line in data.split('\n'):
        line = line.split(',')
        if not (line[-1] != '0' and line[-1] != '1'):
            output = ''
            # add the input values on a line
            for i in range(len(line)-1):
                if column_types[i]: output += '{},'.format(line[i])
                elif line[i] != '': output += '{},'.format(encoder[i][line[i]])
            fout.write(output[:-1] + '\n')

            # if real output
            if column_types[-1]: fout.write('{}\n'.format(line[-1]))
            # else encode class
            elif line[-1] != '': fout.write('{}\n'.format(encoder[len(line)-1][line[-1]]))

    fin.close()
    fout.close()

process_file('datasets/diabetes.csv', 'datasets/diabetes.txt', [9,10],0)
