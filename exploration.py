import pandas as pd

# Datensatz laden
mushroom_data = pd.read_csv("mushrooms.csv")

# Die ersten paar Zeilen des Datensatzes anzeigen
print(mushroom_data.head())

# Informationen über den Datensatz erhalten
print(mushroom_data.info())

# Statistische Zusammenfassung der numerischen Spalten anzeigen
print(mushroom_data.describe())

# Anzahl der eindeutigen Werte in jeder Spalte anzeigen
print(mushroom_data.nunique())
