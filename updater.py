from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

i = 1
j = 1
cardNames = []
quantNames = []
while j < 31:
    cardNames.append('Card' + str(j))
    quantNames.append('Q' + str(j))
    j += 1
columnNames = ['Date', 'Address', 'Name', 'Class', 'Archetype', 'Expansion', 'DeckScore'] + cardNames + quantNames +\
              ['Wins', 'Bucket1', 'Bucket2']

source = requests.get('https://www.heartharena.com/profile/krippers')
soup = BeautifulSoup(source.text, 'lxml')
addList = soup.find('table', class_='arena-runs-table')
df = pd.read_csv('C:\\Users\\chris\\Google Drive\\Python\\KrippHistory.csv')
addresses = df['Address'].tolist()
tableInAdd = addList.find_all('tr', class_='classification')
tableInAdd.reverse()

for link in tableInAdd:
    linkString = 'https://www.heartharena.com' + link['data-href']
    if linkString not in addresses[-20:]:
        # Date
        date = link.find('td', class_='created-at').text
        dateObj = datetime.strptime(date, '%b %d, %Y')
        currentDeck = [dateObj.strftime('%Y-%m-%d')]
        # Address
        currentDeck.append(linkString)
        # Deck Name
        currentDeck.append(linkString[38:44])
        # Break Out the New Soup
        deckSource = requests.get(linkString)
        newSoup = BeautifulSoup(deckSource.text, 'lxml')
        cardList = newSoup.find('ul', class_='deckList')
        gameList = newSoup.find('ul', class_='matches-list')
        # Class
        hsClass = newSoup.title.text.split()[0]
        currentDeck.append(hsClass)
        # Archetype
        archetype = newSoup.find('header', class_='deck-archetype-name').find('span')
        archetype = str(archetype)
        deckType = archetype[6:]
        currentDeck.append(deckType.split('<br/>')[0])
        # Expansion
        currentDeck.append('RoS')
        # Deck Score
        currentDeck.append(newSoup.find('section', class_='arenaDeckTierscore').find('span').text)
        # Cards in Deck
        for card in cardList.find_all('span', class_='name'):
            currentDeck.append(card.text)
        more = 37-len(currentDeck)
        extra = [None] * more
        currentDeck = currentDeck + extra
        # Quantity of Each Card in Deck
        for card in cardList.find_all('span', class_='quantity'):
            currentDeck.append(card.text)
        currentDeck = currentDeck + extra
        wins = int(newSoup.find('span', class_='wins').text)
        # Wins
        currentDeck.append(wins)
        # Bucket 1
        if wins > 8:
            bucket1 = 2
        elif wins > 4:
            bucket1 = 1
        else:
            bucket1 = 0
        currentDeck.append(bucket1)
        # Bucket 2
        if wins > 6:
            bucket2 = 'High'
        elif wins > 0:
            bucket2 = 'Low'
        else:
            bucket2 = 'Zero'
        currentDeck.append(bucket2)
        # Saving all info to a new csv
        newDf = pd.DataFrame(currentDeck).T
        newDf.columns = columnNames
        df = df.append(newDf, ignore_index=True)
        print('Deck', i, 'added.')
        i += 1

df.to_csv('C:\\Users\\chris\\Google Drive\\Python\\KrippHistory.csv', index=False)
