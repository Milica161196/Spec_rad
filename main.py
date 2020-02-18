import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Activation


(XTrain, YTrain), (XTest, YTest) = imdb.load_data(num_words=10000)

#ovaj dio mijenja odnos trening-test uzoraka na 80%-20%
data = np.concatenate((XTrain, XTest), axis=0)
targets = np.concatenate((YTrain, YTest), axis=0)
XTest = data[:10000]
YTest = targets[:10000]
XTrain = data[10000:]
YTrain = targets[10000:]

WordIndex = imdb.get_word_index()

for value in XTrain[0]:
    for (key1, value1) in WordIndex.items():
        if value - 3 == value1:
            print(key1 + ' ', end=' ')
print()
#pripremamo review-e za ucitavanje u tenzor tako sto ih
#postavljamo na istu duzinu
XTrain = pad_sequences(XTrain, maxlen=10000)
XTest = pad_sequences(XTest, maxlen=10000)


#ovdje definisemo linearni stek mreznih slojeva koji cine model
#kao konstruktor mreze na koji onda dodajemo slojeve
RNNModel = Sequential()

#prvo ide Embedding layer koji uzima 2D tensor i vrace 3D tensor
RNNModel.add(Embedding(10000, 32, input_length=10000))

#dodajemo sloj rekurentne mreze gdje se izlaz hrani na ulaz
RNNModel.add(SimpleRNN(32, input_shape=(10000, 10000), return_sequences=False))

#sloj dense nam vrace jednodimenzionalni izlaz
RNNModel.add(Dense(1))

#aktivacioni sloj postavlja output na vrijednosti izmedju 0 i 1
RNNModel.add(Activation('sigmoid'))

#proces ucenja
RNNModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

RNNModel.fit(XTrain, YTrain, validation_data=(XTest, YTest), epochs=2, batch_size=64, verbose=2)

scores = RNNModel.evaluate(XTest, YTest, verbose=2)

print("Accuracy: %.2f%%" % (scores[1]*100))

'''
Filmski opisi su kodirani tako sto je svakoj rijeci u svim kritikama izracunata ucestalost (popularnost)
pa su im na osnovu toga dodijeljeni indeksi kojima su onda i rijeci zamijenjene u tekstu.
(npr. indeks 3 bi imala treca najpopularnija rijec u dataset-u)

Opis parametara:
    num_words - odredjujemo koliko prvih najpopularnijih rijeci da uzimamo
    input_length - ovdje odredjujemo duzinu pojedinacnih kritika
    activation - tip funkcije kojim definisemo izlaz cvorova mreze
    loss - funkcija kojom optimizujemo mrezu
    epoch - broj epoha
    batch_size - broj trening uzoraka koji uzimamo prije nego se azuriraju parametri mreze

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 80.72%

num_words=1000; input_length=50; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 76.89%

num_words=1000; input_length=30; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 72.21%

num_words=1000; input_length=10; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 67.94%

num_words=100; input_length=10; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 59.40%

num_words=10; input_length=10; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 53.34%

Sa smanjivanjem duzine sekvenci (tekstova) koje uzimamo smanjuje se i preciznost.(input_length)
Isti efekat ima i smanjivanj broja najcescih rijeci.(num_words)

################################

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=2, batch_size=64
Accuracy: 80.50%

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 81.57%

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=10, batch_size=64
Accuracy: 79.84%

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=20, batch_size=64
Accuracy: 72.66%

num_words=1000; input_length=100; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=50, batch_size=64
Accuracy: 74.98%

Izgleda da povecanje broja epoha sa ovakvim ostalim parametrima ne donosi znacajno poboljsanje, cak naprotiv.

################################

num_words=100; input_length=20; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=3, batch_size=64
Accuracy: 61.98%

num_words=100; input_length=20; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=5, batch_size=64
Accuracy: 61.34%

num_words=100; input_length=20; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=4, batch_size=64
Accuracy: 61.76%

num_words=100; input_length=20; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=10, batch_size=64
Accuracy: 61.27%

####################################

num_words=1000; input_length=100; aktivacija='tanh'; loss='binary_crossentropy'; epochs=3, batch_size=64
Accuracy: 70.37%

num_words=1000; input_length=100; aktivacija='tanh'; loss='binary_crossentropy'; epochs=6, batch_size=64
Accuracy: 71.46%

Drugi tip aktivacione funkcije nam nije donio poboljsanje.

#################################

#Sad znatno povecavamo num_words i input_length
num_words=10000; input_length=200; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=6, batch_size=64
Accuracy: 82.83%

num_words=10000; input_length=200; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=3, batch_size=64
Accuracy: 78.70%

################################
num_words=10000; input_length=200; aktivacija='sigmoid'; loss='binary_crossentropy'; epochs=2, batch_size=10
Accuracy: 78.70%

'''

