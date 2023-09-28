from sklearn.model_selection import train_test_split

from tensorflow import keras
import time
import numpy as np
import pandas as pd
from pandas import read_excel
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.feature_extraction import DictVectorizer
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Activation, InputLayer, \
    Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

import difflib

# Какие то гиперпараметры
CLASSES_NUM = 4
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
    0: 'etc'
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


# Что в самой нейронке
class BibliographicList(keras.Model):
    def __init__(
            self, hidden_neurons=256, output_dim=100,
            filters=100, kernel_size=2, pool_size=2,
            voc_size=VOCAB_SIZE, embed_dim=MAX_SIZE_OF_LINE
    ):
        super().__init__()
        self.embedding = keras.layers.Embedding(
            input_dim=voc_size, output_dim=10, input_length=embed_dim)

        self.sequential = keras.Sequential([
            Conv1D(filters, kernel_size, activation='relu'),
            MaxPooling1D(pool_size),
            LSTM(hidden_neurons)
        ])

        self.dropout1 = layers.Dropout(0.1)
        self.layer1 = Dense(32, activation='relu')
        self.layer2 = Dense(NUM_OF_CLASSES, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.sequential(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)

        return x


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


# Пред. варианты преобразовния строки
def from_string_to_num(source, classes):
    # d = words_coding()
    # d_list = list(words_coding().values())
    # print(source)
    # old_source = source.split()
    # source = source.replace(' ', '')
    # print(source)
    # for k, tp in enumerate(classes):
    #     n = len(tp.split())
    #     tp = tp.replace(' ', '')
    #     print(tp)
    #     source = source.replace(tp, str(k + 1) * n, 1)
    #
    # for word in old_source:
    #     if word in source:
    #         source = source.replace(word, '0')
    #
    # source = source.replace('.', '')
    # source = source.replace(',', '')
    # print(source)
    # new_source = str()
    #
    # for index, cl in enumerate(classes):
    #     for sub_cl in cl.split():
    #         for word in source.split():
    #             print(sub_cl, word)
    #             if word == sub_cl:
    #                 new_source = new_source + str(index + 1)
    #                 break
    old_source = source.split()
    print(source)
    source = source.replace(' ', '')
    for k, cl in enumerate(classes):
        num_words = len(cl.split())
        cl = cl.replace(' ', '')
        length = len(cl)
        for index in range(len(source)):
            print(cl, source[index: index + length], source)
            if source[index: index + length] == cl:
                source = source[:index] + str(k + 1) * num_words + source[index + length:]
                break
    print(sorted(old_source, key=len)[::-1])
    for old_word in sorted(old_source, key=len)[::-1]:
        str1 = old_word
        str2 = old_word
        k = len(old_word) // 2
        while len(str1) != k:
            print(str1, str2, source)
            if len(str1) == 0 or len(str2) == 0:
                break
            if str1 in source:
                source = source.replace(str1, '0', 1)
                break
            elif str2 in source:
                source = source.replace(str2, '0', 1)
                break
            else:
                str1 = str1[1:]
                str2 = str2[:len(str2) - 1]

    print('???????', source, '????????')
    source = source.replace('.', '')
    source = source.replace(',', '')
    source = source.replace(':', '')
    source = source.replace(';', '')

    if '.' in source or ',' in source:
        print()
        print(source)
        exit()

    source = check(source)
    source = list(source)

    if len(old_source) != len(source):
        print('\t'.join(old_source))
        print('\t'.join(source))
        for i in range(max(len(old_source), len(source))):
            try:
                print(old_source[i], source[i])
            except:
                try:
                    print(old_source[i], -1)
                except:
                    print(-1, source[i])
        print(len(old_source), len(source))
        exit()

    # print('True')
    # print(new_source)
    source = [int(i) for i in source]
    return tf.cast(source, tf.float32)


# Здесь экселевские данные разделяются на: строку и классы
def map_record_to_training_data(record):
    record = record.split('\t')
    source = record[0]
    classes = record[1:]
    return source, classes[:CLASSES_NUM]  # CLASSES_NUM - кол-во классов на которые нейронка будет распределять слова в строке


# Функция запуска нейронки для отработки одной строки
def predict_nn(data, model):
    output = model.predict(tf.cast(data, tf.float32))
    prediction = np.argmax(output, axis=-1)
    print(prediction)
    return prediction


# Функция начала работы нейронки для данных
# Она принимает строки
def start_nn(data, model):
    data_list = list()
    for line in data:
        print(line)
        dictionary_of_data = {'String': line,
                              'After_nn': predict_nn(from_sentence_to_nums(line), model)}

        dictionary_of_types = dict()
        for i in range(len(CODING)):
            dictionary_of_types[CODING[i]] = list()

        for num in range(len(CODING)):
            prediction = dictionary_of_data['After_nn']
            for index, word in enumerate(dictionary_of_data['String'].split()):
                if prediction[index] == num:
                    dictionary_of_types[CODING[num]].append(word)

        print(dictionary_of_types)
        dictionary_of_data['Distribution'] = dictionary_of_types
        data_list.append(dictionary_of_data)

    return data_list

# Открытие файла (без преобразования данных для нейронки)
df = pd.read_excel('C:/Users/okudz/Desktop/mirea/1000_tests.xlsx')


# Преобразование данных для нейронки (вариант сделан для последующих изменений, если таковы нужны)
train_dataset = (
    concatenation(df).map(map_record_to_training_data)
    .map(lambda x: (from_sentence_to_nums(x[0]), biblio_matrix(x[0], x[1])))
)

# Экземпляр класса
biblio_model = BibliographicList()

# Задание параметров для обучения нейронки
biblio_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


X = [i[0] for i in train_dataset]
Y = [to_categorical(i[1], NUM_OF_CLASSES) for i in train_dataset]
for_predict = X[-1]

new_x = list()
new_y = list()

for l in X:
    for i in l:
        new_x.append(i)

for l in Y:
    for i in l:
        new_y.append(i)


# Разделение данных для обучения (0.7 = 7 к 3 = на 70% обучается, на 30% тестируется)
x_train, x_test, y_train, y_test = train_test_split(new_x, new_y, train_size=0.7, random_state=42)

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)

# Метод для обучения нейронки:
# epochs - кол-во циклов обучения нейронки
biblio_model.fit(x_train, y_train, epochs=50)

# Тестирование данных
biblio_model.evaluate(x_test, y_test)

# Сохранение модели нейронной сети
biblio_model.save('какое_то_название_нейронки')

# Открытие нейронки и запись в какую то переменную
model = keras.models.load_model('какое_то_название_нейронки')

print(start_nn([
    'Zoppi G., Luciano A., Cinquetti M. et al. Respiratory quotient changes in full '
    'term newborn infants within 30 hours from birth before start of milk feeding // '
    'Eur. J. Clin. Nutr. – 1998. – Vol. 52, № 5. – P. 360–362. doi: 10.1038/sj.ejcn.1600564.',
    'Zoremba M., Kalmus G., Dette F. et al. Effect of intra-operative pressure support '
    'vs pressure controlled ventilation on oxygenation and lung function in moderately '
    'obese adults // Anaesthesia. – 2010. – Vol. 65, № 2. – P. 124–129. doi: 10.1111/j.'
    '1365-2044.2009.06187.x.'],
    model))
