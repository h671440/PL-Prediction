import pandas as pd
from sklearn.preprocessing import LabelEncoder

#laster inn data
df = pd.read_csv("../data/E0-2.csv")

#Håndterer data som ikke er fylt inn, erstatter data med medianen for kolonnen. 
df.fillna(df.mean(), inplace= True)

#gjør om hjemme og bortelag til numeriske verdier
le = LabelEncoder()
df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
df['AwayTeam'] = le.fit_transform(df['AwayTeam'])

#konvertere data til numeriske verdier H = hjemmeseier, D = uavgjort, A = borteseier
# 1 = seier, 0 = uavgjort, -1 = borteseier /tap

df['FTR'] = df['FTR'].map({'H':1, 'D': 0, 'A': -1 })

#fikse målforskejll og diverse
df['GoalDifference'] = df['FTHG'] - df['FTAG']
df['ShotsEffiency_Home'] = df['HST'] / df['HS']
df['ShotsEffiency_Away'] = df['AST'] / df['AS']

#dropp rader uten data
df.dropna(inplace=True)

#lagre den nye prosseserte dataen 
df.to_csv("../data/processed_data.csv", index = False)

print("Data prepocessing fullført og lagret som '../data/processed_data.csv'")





