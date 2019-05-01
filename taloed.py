from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import talos
from tensorflow.keras import layers, callbacks
import os

print(tf.VERSION)
print(tf.keras.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Preparing dictionaries to convert data into integers. Later they will be turned to one-hots.

classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
             'Warlock': 7, 'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

df = pd.read_csv('/home/chris/Desktop/KrippDateless.csv', sep=',', header=None)
df = df.sample(frac=1)

preNump = []
preLabels = []
best = 0

for row in df.itertuples():

    # Turning classes to hots.
    hotNum = classDict[row[1]]
    classTens = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classTens[hotNum] = 1

    # Normalizing the deck scores before adding them to the tensor.
    deckScore = ((row[2]-50)/30)

    # Turn deck type into a hot.
    deckHot = [0]*6
    deckHot[deckType[row[3]]] = 1

    # Turn expansion into a hot.
    expansHot = [0]*12
    expansHot[expansion[row[4]]] = 1

    # Combine data them into a single tensor.
    classTens.append(deckScore)
    combinedTens = classTens + deckHot + expansHot

    # Add that tensor to the list of tensors.
    preNump.append(combinedTens)

    # Convert that label to a hot and add to the list of labels.
    labelHot = [0, 0, 0]
    labelHot[row[5]] = 1
    preLabels.append(labelHot)

splitnum = int(df.shape[0]*.8)
traindata = np.array(preNump[:splitnum])
evaldata = np.array(preNump[splitnum:])
trainlabels = np.array(preLabels[:splitnum])
evallabels = np.array(preLabels[splitnum:])
p = {'layer1': [12, 24, 48], 'batch_size': [10, 20, 40]}


def runnn(x_train, y_train, x_val, y_val, params):

    model = tf.keras.Sequential([layers.Dense(params['layer1'], kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                              activation='relu', input_shape=(28,)),
                                 layers.Dropout(.3),
                                 layers.Dense(100, activation='relu'),
                                 layers.Dropout(.2),
                                 layers.Dense(3, activation='softmax')])

    optimizer = tf.keras.optimizers.Adam(lr=0.003)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    out = model.fit(x=x_train, y=y_train, batch_size=params['batch_size'], epochs=100, verbose=0,
                        validation_data=(x_val, y_val), shuffle=True)

    return out, model


talos.Scan(x=traindata, y=trainlabels, x_val=evaldata, y_val=evallabels, model=runnn, params=p)
