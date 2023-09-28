from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import difflib
import time


# Какие то гиперпараметры
ERROR = '!@#$%^&*())(*&^%$#@!'
NUM_OF_CLASSES = 11
MAX_SIZE_OF_LINE = 10
VOCAB_SIZE = 1000
OUTPUT_DIM = 10

# Как что кодируется (классы)
CODING = {
    1: 'Author',
    2: 'Title',
    3: 'Journal',
    4: 'Year',
    5: 'Volume',
    6: 'Release',
    7: 'Start page',
    8: 'End page',
    9: 'Link',
    0: 'etc',

}


def similar(seq1, seq2):
    return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio()

# Функция преобразования слова в строке в набор параметров
def from_sentence_to_nums(string):
    list_made_of_attributes = list()
    sent = dict()

    for index, word in enumerate(string.split()):
        attributes = dict()
        if len(word) == 1 and word in ['-', '//']:
            attributes['Symbol'] = 0
            attributes['Uppercase'] = 0
            if word == '//':
                attributes['Separators'] = 2
            elif word == '-':
                attributes['Separators'] = 1
            else:
                attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        elif word.isalpha():
            attributes['Symbol'] = 1
            if word[0].isupper():
                attributes['Uppercase'] = 2
            elif word[0].islower():
                attributes['Uppercase'] = 1
            else:
                attributes['Uppercase'] = 0
            attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        elif word.isdigit():
            attributes['Symbol'] = 2
            attributes['Uppercase'] = 0
            attributes['Separators'] = 0
            attributes['DotOrComma'] = 0
            attributes['Initial'] = 0

        else:
            attributes['Symbol'] = 3
            attributes['Uppercase'] = 0
            if '//' in word:
                attributes['Separators'] = 2
            elif '-' in word:
                attributes['Separators'] = 1
            else:
                attributes['Separators'] = 0

            if '.' in word:
                attributes['DotOrComma'] = 1
            elif ',' in word:
                attributes['DotOrComma'] = 2
            else:
                attributes['DotOrComma'] = 0

            attributes['Initial'] = 1

        list_made_of_attributes.append(attributes)
    new_list_made_of_attributes = list()

    for i, attributes in enumerate(list_made_of_attributes):
        new_attributes = dict()
        if i == 0:
            for att in attributes.keys():
                new_attributes['Prev' + att] = 0
        else:
            for att in attributes.keys():
                new_attributes['Prev' + att] = list_made_of_attributes[i - 1][att]

        for att in attributes.keys():
            new_attributes[att] = list_made_of_attributes[i][att]

        if i == (len(list_made_of_attributes) - 1):
            for att in attributes.keys():
                new_attributes['Next' + att] = 0
        else:
            for att in attributes.keys():
                new_attributes['Next' + att] = list_made_of_attributes[i + 1][att]

        new_list_made_of_attributes.append(list(new_attributes.values()))
    return new_list_made_of_attributes


def concatenation(data):
    heads = data.keys()
    new_data = data[heads[0]]

    for head in heads[1:]:
        new_data = new_data + '\t' + data[head].apply(str)

    return new_data

# Функция кодирования данных из строкового типа в численный
def biblio_matrix(source, classes):
    lst = list()
    source_one_string = source[:].replace(' ', '')
    source_splitted = source[:].split()
    classes = [cl.split() for k, cl in enumerate(classes)]

    matrix = [[0 for j in range(len(source_splitted))] for i in range(len(classes))]

    for k in range(len(classes)):
        try:
            num_words = len(classes[k])
            probabilities = [similar(classes[k][0], string) for string in source_splitted]
            index = probabilities.index(max(probabilities))

            for i in range(num_words):
                matrix[k][index + i] = k + 1
        except Exception as e:
            continue

    for i in range(len(matrix[0])):
        matrix[0][i] = max([j[i] for j in matrix])

    return tf.cast(matrix[0], tf.float32)

# Здесь экселевские данные разделяются на: строку и классы
def map_record_to_training_data(record):
    record = record.split('\t')
    source = record[0]
    classes = record[1:]
    print(classes)
    return source, classes


def one_column(dataframe):
    return dataframe['source']

# Функция запуска нейронки для отработки одной строки
def predict_nn(data, model):
    preditcion = list()
    try:
        output = model.predict(tf.cast(data, tf.float32))
        prediction = np.argmax(output, axis=-1)
    except Exception as e:
        prediction = [ERROR, str(e)]
    return prediction

# Функция начала работы нейронки для данных
# Она принимает строки
def start_nn(data, model):
    data_list = list()
    for line in data:
        dictionary_of_data = {'String': line,
                              'After_nn': predict_nn(from_sentence_to_nums(line), model),
                              'ERROR': False}
        dictionary_of_types = dict()

        for i in range(len(CODING)):
            dictionary_of_types[CODING[i]] = list()

        if dictionary_of_data['After_nn'][0] == ERROR:
            for i in range(len(dictionary_of_data['After_nn'])):
                try:
                    dictionary_of_types[CODING[i + 1]].append(dictionary_of_data['After_nn'][i])
                except:
                    dictionary_of_types[CODING[0]].append(dictionary_of_data['After_nn'][i])
            dictionary_of_data['ERROR'] = True
        else:
            for num in range(len(CODING)):
                prediction = dictionary_of_data['After_nn']
                for index, word in enumerate(dictionary_of_data['String'].split()):
                    if prediction[index] == num:
                        dictionary_of_types[CODING[num]].append(word)

        dictionary_of_data['Distribution'] = dictionary_of_types
        data_list.append(dictionary_of_data)

    # for d in data_list:
    #     print(d['String'])
    #     for key in d['Distribution'].keys():
    #         print(key, d['Distribution'][key])

    return data_list

def print_in_file(data):
    dict_to_df = {'String': [data[i]['String'] for i in range(len(data))]}

    for i in range(len(CODING)):
        dict_to_df[CODING[i]] = list()

    for num in range(len(data)):
        for i in range(len(CODING)):
            lst = data[num]['Distribution'][CODING[i]]
            new_string = ' '.join(lst) if lst else '-'
            dict_to_df[CODING[i]].append(new_string)
            # if data[num]['ERROR']:
            #     dict_to_df[CODING[0]]

    return pd.DataFrame(dict_to_df)

# Открытие файла (без преобразования данных для нейронки)
df = pd.read_csv('C:/Users/1/Downloads/nn_library_list-main/twoMillion.csv')

sources = one_column(df).tolist()

model = keras.models.load_model('C:/Users/1/Downloads/nn_library_list-main/model_first4')
df = print_in_file(start_nn(sources[85815:86815], model))

# Данные импортируются в файл 
# mode - как данные импортируются
# mode='b' - файл чистится, а после данные закидываются в файл
# mode='a' - данные импортируются в непустой файл
df.to_csv('twoMillionCSV.csv', mode='a', header=False, index=False)

# Удобный вывод
print('Обработано: ', 1000)
t = time.process_time()
seconds = t
hours = seconds // 3600
minutes = (seconds - hours * 3600) // 60 

print('Затраченное время:', hours, 'ч ', minutes, 'мин ', seconds - hours * 3600 - minutes * 60, 'сек')
print(end='\n\n\n\n')


for i in range(87815, len(sources), 1000):
    df = print_in_file(start_nn(sources[i - 1000: i], model))

    # saving the DataFrame as a CSV file
    df.to_csv('twoMillionCSV.csv', mode='a', header=False, index=False)
    print('Обработано: ', i)
    seconds = time.process_time() - t
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60 

    print('Затраченное время:', hours, 'ч ', minutes, 'мин ', seconds - hours * 3600 - minutes * 60, 'сек')
    print(end='\n\n\n\n')