from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd

print(tf.VERSION)
print(tf.keras.__version__)

# We're doing to use this to load a model and predict.
model = tf.keras.models.load_model('/home/chris/Desktop/RoS5279.h5')

# Preparing dictionaries to convert data into integers. Later they will be turned to one-hots.
classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
             'Warlock': 7, 'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

columnNames = []
listOLists = []

for hsClass in classDict:
    for dt in deckType:
        score = 50
        column = dt + ' ' + hsClass
        columnNames.append(column)
        pre2 = []
        while score < 81:
            #     [Class,  Score,  Deck Type,  Expansion]
            row = [hsClass, score, dt, 'Rumble']

            # Turning classes to hots.
            hotNum = classDict[row[0]]
            classTens = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            classTens[hotNum] = 1

            # Normalizing the deck scores before adding them to the tensor.
            deckScore = ((row[1] - 50) / 30)

            # Turn deck type into a hot.
            deckHot = [0] * 6
            deckHot[deckType[row[2]]] = 1

            # Turn expansion into a hot.
            expansHot = [0] * 12
            expansHot[expansion[row[3]]] = 1

            # Combine data them into a single tensor.
            classTens.append(deckScore)
            combinedTens = classTens + deckHot + expansHot

            predicTens = [[combinedTens]]
            prediction = model.predict(predicTens)

            score += 1
            pre2.append(prediction[0][1])
        listOLists.append(pre2)
df = pd.DataFrame(listOLists)
indexNames = list(range(50, 81))
df.index = columnNames
df.columns = indexNames
print(df)
df.to_csv('/home/chris/Desktop/maddoc.csv')
