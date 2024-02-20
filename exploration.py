import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Datensatz laden
mushroom_data = pd.read_csv("mushrooms.csv")

# Die ersten paar Zeilen des Datensatzes anzeigen
print(mushroom_data.head())

# Daten vorbereiten
label_encoder = LabelEncoder()
for column in mushroom_data.columns:
    mushroom_data[column] = label_encoder.fit_transform(mushroom_data[column])

# Aufteilung in Features (X) und Zielvariable (y)
X = mushroom_data.drop('class', axis=1)
y = mushroom_data['class']

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell erstellen (hier verwenden wir Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test)

# Evaluierung des Modells
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Ergebnisse anzeigen
print(f'Genauigkeit: {accuracy}')
print('\nKonfusionsmatrix:')
print(conf_matrix)
print('\nKlassifikationsbericht:')
print(class_report)

# Heatmap für die Korrelation der Merkmale
plt.figure(figsize=(12, 10))
sns.heatmap(mushroom_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korrelation der Merkmale')
plt.show()