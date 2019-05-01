import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

df = pd.read_csv('/home/chris/Desktop/KrippHistory.csv')

# Create the unique card dictionary
i = 1
cards = []
preNump = []
preLabels = []
cardDic = {}
while i < 31:
    cardNum = 'Card' + str(i)
    cards = cards + df[cardNum].tolist()
    i += 1
print('Total Cards:', len(cards))
noDups = list(set(cards))
noDupsLen = len(noDups)
print('Unique Cards', noDupsLen)

j = 0
for card in noDups:
    cardDic[card] = j
    j += 1

# The other dictionaries
classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6, 'Warlock': 7,
             'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'GvG': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

# Create the tensors
for row in df.itertuples():

    # Classes
    classTens = [0] * 9
    classTens[classDict[row[4]]] = 1

    # Archetype
    archTens = [0] * 6
    archTens[deckType[row[5]]] = 1

    # Expansion
    expanTens = [0] * 12
    expanTens[expansion[row[6]]] = 1

    # Normalizing the deck scores before adding them to the tensor.
    deckScore = ((row[7] - 50) / 30)

    # Cards
    cardTens = [0] * noDupsLen
    k = 8
    while k < 38:
        if row[k] is not np.nan:
            cardTens[cardDic[row[k]]] = row[k+30]
        k += 1

    # Combine data into a single tensor.
    combinedTens = classTens + archTens + expanTens
    combinedTens.append(deckScore)
    combinedTens = combinedTens + cardTens

    # Add that tensor to the list of tensors.
    preNump.append(combinedTens)

    # Convert labels to hots and add to the list of labels.
    labelHot = [0, 0, 0]
    labelHot[row[69]] = 1
    preLabels.append(labelHot)

print('Input', len(combinedTens))
trainData = np.array(preNump)
trainLabels = np.array(preLabels)

count = 0
while count < 10:
    graph = tf.Graph()
    with tf.Session(graph=graph):
        model = tf.keras.Sequential([layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(.001),
                                                  activation='relu', input_shape=(len(combinedTens),)),
                                     layers.Dropout(0.5),
                                     layers.Dense(100, activation='relu'),
                                     layers.Dropout(0.3),
                                     layers.Dense(3, activation='softmax')])

        optimizer = tf.keras.optimizers.Adam(lr=0.00001)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x=trainData, y=trainLabels, batch_size=16, epochs=20, validation_split=0.2, shuffle=True)
    count += 1
