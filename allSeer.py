from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pickle
from bs4 import BeautifulSoup
import requests
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load the model, card dictionary, and address
simpleModel = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\RoS5279.h5')
oModel = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\OModel5316.h5')
oPickleIn = open('C:\\Users\\chris\\Google Drive\\Python\\ODict5316.pkl', 'rb')
oCardDict = pickle.load(oPickleIn)
nModel = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\NModel6151.h5')
nPickleIn = open('C:\\Users\\chris\\Google Drive\\Python\\NDict6151.pkl', 'rb')
nCardDict = pickle.load(nPickleIn)
wModel = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\WModel9306.h5')
wPickleIn = open('C:\\Users\\chris\\Google Drive\\Python\\WDict9306.pkl', 'rb')
wCardDict = pickle.load(wPickleIn)
wCModel = tf.keras.models.load_model('C:\\Users\\chris\\Google Drive\\Python\\WCModel1688.h5')
wCPickleIn = open('C:\\Users\\chris\\Google Drive\\Python\\WCDict1688.pkl', 'rb')
wCCardDict = pickle.load(wCPickleIn)
address = 'https://www.heartharena.com/arena-run/49xur1'
lettuce = 1381

# Preparing dictionaries to convert data into integers. Later they will be turned to one-hots.
classDict = {'Druid': 0, 'Hunter': 1, 'Mage': 2, 'Paladin': 3, 'Priest': 4, 'Rogue': 5, 'Shaman': 6,
             'Warlock': 7, 'Warrior': 8}
deckType = {'Aggro-Control': 0, 'Attrition': 1, 'Classic Aggro': 2, 'Classic Control': 3, 'Mid-Range': 4, 'Tempo': 5}
expansion = {'Vanilla': 0, 'BRM': 1, 'WOG': 2, 'Kara': 3, 'MSG': 4, 'Ungoro': 5, 'KFT': 6, 'KnC': 7, 'Woods': 8,
             'Boomsday': 9, 'Rumble': 10, 'RoS': 11}

# Load deck info from address
source = requests.get(address)
newSoup = BeautifulSoup(source.text, 'lxml')
cardList = newSoup.find('ul', class_='deckList')
gameList = newSoup.find('ul', class_='matches-list')

# Class
hsClass = newSoup.title.text.split()[0]
classTens = [0] * 9
classTens[classDict[hsClass]] = 1
simpleClassTens = [0] * 9
simpleClassTens[classDict[hsClass]] = 1

# Archetype
archetype = newSoup.find('header', class_='deck-archetype-name').find('span')
archetype = str(archetype)
archetype = archetype[6:]
archetype = archetype.split('<br/>')[0]
archTens = [0] * 6
archTens[deckType[archetype]] = 1

# Expansion
expanTens = [0] * 12
expanTens[expansion['RoS']] = 1

# Deck Score
score = newSoup.find('section', class_='arenaDeckTierscore').find('span').text
deckScore = (float(score)-50)/30

# Cards in Deck
deckList = []
for card in cardList.find_all('span', class_='name'):
    deckList.append(card.text)

# Quantity of Each Card in Deck
quantList = []
for card in cardList.find_all('span', class_='quantity'):
    quantList.append(card.text)

# Form the card tensor
oCardTens = [0] * len(oCardDict)
nCardTens = [0] * len(nCardDict)
wCardTens = [0] * len(wCardDict)
wCCardTens = [0] * len(wCCardDict)
i = 0
while i < len(deckList):
    try:
        oCardTens[oCardDict[deckList[i]]] = quantList[i]
        nCardTens[nCardDict[deckList[i]]] = quantList[i]
        wCardTens[wCardDict[deckList[i]]] = quantList[i]
        wCCardTens[wCCardDict[deckList[i]]] = quantList[i]
    except KeyError:
        print('One card missing.')
        pass
    i += 1

# Combine data them into a single tensor.
simpleClassTens.append(deckScore)
simpleTens = simpleClassTens + archTens + expanTens

oCombinedTens = classTens + archTens + expanTens
oCombinedTens.append(deckScore)
oCombinedTens = oCombinedTens + oCardTens

nCombinedTens = classTens + archTens + expanTens
nCombinedTens.append(deckScore)
nCombinedTens = nCombinedTens + nCardTens

wCombinedTens = classTens + archTens + expanTens
wCombinedTens.append(deckScore)
wCombinedTens = wCombinedTens + wCardTens

wCCombinedTens = classTens + archTens + expanTens
wCCombinedTens.append(deckScore)
wCCombinedTens = wCCombinedTens + wCCardTens

simplePredicTens = [[simpleTens]]
simplePrediction = simpleModel.predict(simplePredicTens)
oPredicTens = [[oCombinedTens]]
oPrediction = oModel.predict(oPredicTens)
nPredicTens = [[nCombinedTens]]
nPrediction = nModel.predict(nPredicTens)
wPredicTens = [[wCombinedTens]]
wPrediction = wModel.predict(wPredicTens)
wCPredicTens = [[wCCombinedTens]]
wCPrediction = wCModel.predict(wCPredicTens)

sBet = (max(simplePrediction[0]) - .5) * 2 * lettuce
oBet = (max(oPrediction[0]) - .5) * 2 * lettuce
nBet = (max(nPrediction[0]) - .5) * 2 * lettuce

counter = 1
predictSum = 0
while counter < 13:
    predictSum = predictSum + wCPrediction[0][counter]*counter
    counter += 1

print(archetype, hsClass, score)
print(f'Logistic Prediction: {wPrediction[0][0]:.2f} wins.')
print(f'Category Prediction: {predictSum:.2f} wins.')
print('Win Probabilities:')
print(' 0:', wCPrediction[0][0])
print(' 1:', wCPrediction[0][1])
print(' 2:', wCPrediction[0][2])
print(' 3:', wCPrediction[0][3])
print(' 4:', wCPrediction[0][4])
print(' 5:', wCPrediction[0][5])
print(' 6:', wCPrediction[0][6])
print(' 7:', wCPrediction[0][7])
print(' 8:', wCPrediction[0][8])
print(' 9:', wCPrediction[0][9])
print('10:', wCPrediction[0][10])
print('11:', wCPrediction[0][11])
print('12:', wCPrediction[0][12])
print('\n 1-4:', wCPrediction[0][0] + wCPrediction[0][1] + wCPrediction[0][2] + wCPrediction[0][3] + wCPrediction[0][4])
print(' 5-8:', wCPrediction[0][5] + wCPrediction[0][6] + wCPrediction[0][7] + wCPrediction[0][8])
print('9-12:', wCPrediction[0][9] + wCPrediction[0][10] + wCPrediction[0][11] + wCPrediction[0][12])
print('\n 1-6:', wCPrediction[0][1] + wCPrediction[0][2] + wCPrediction[0][3] + wCPrediction[0][4] + wCPrediction[0][5]
      + wCPrediction[0][6])
print('7-12:', wCPrediction[0][7] + wCPrediction[0][8] + wCPrediction[0][9] + wCPrediction[0][10] + wCPrediction[0][11]
      + wCPrediction[0][12])
print('\nSimple MechaKripp prediction:', simplePrediction[0])
print('Bet', int(sBet), 'lettuce.')
print('\nMechaKripp paradigm prediction:', oPrediction[0])
print('Bet', int(oBet), 'lettuce.')
print('\nStreamLabs paradigm prediction:', nPrediction[0])
print('Bet', int(nBet), 'lettuce.')
