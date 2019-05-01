from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime

source = requests.get('https://www.heartharena.com/profile/krippers')
soup = BeautifulSoup(source.text, 'lxml')
addList = soup.find('table', class_='arena-runs-table')
df = pd.read_csv('/home/chris/Desktop/datedURLs.csv')
addresses = df['Address'].tolist()
tableInAdd = addList.find_all('tr', class_='classification')
tableInAdd.reverse()

newAdds = []

for link in tableInAdd:
    linkString = 'https://www.heartharena.com' + link['data-href']
    if linkString not in addresses[-20:]:
        # Date
        date = link.find('td', class_='created-at').text
        dateObj = datetime.strptime(date, '%b %d, %Y')
        newDate = dateObj.strftime('%Y-%m-%d')
        df2 = pd.DataFrame([newDate, linkString]).T
        df2.columns = ['Date', 'Address']
        df = df.append(df2, ignore_index=True)
        print(newDate, linkString, "added.")
df.to_csv('/home/chris/Desktop/datedURLs.csv', index=False)
