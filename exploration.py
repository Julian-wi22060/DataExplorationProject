import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import chi2, f_classif, SelectKBest, RFE

# Datensatz laden
path = "mushrooms.csv"
mushroom_data = pd.read_csv(path)

'''
# Die ersten paar Zeilen des Datensatzes anzeigen
print(mushroom_data.head())

# Die Spaltennamen des Datensatzes anzeigen
print(mushroom_data.columns)

# Feature-Importance auf Grundlage eines Decision-Tree

# Paarplot für alle Merkmale
selected_features = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises']
sns.pairplot(data=mushroom_data, vars=selected_features, palette='Set1')
plt.suptitle('Pairplot für ausgewählte Merkmale', y=1.02)
plt.show()

X=mushroom_data.drop('class', axis=1) 
y=mushroom_data['class'] 
X.head()

Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])

Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=100)
print(f"Number of samples for training set:{X_train.shape}")
print(f"Number of samples for test set:{X_test.shape}")

warnings.filterwarnings("ignore")
logreg = LogisticRegression()

# Feature selection using Recursive Feature Elimination (RFE)
rfe = RFE(estimator=logreg, n_features_to_select=10, step=1)
rfe.fit(X_train, Y_train)

selected_features = X.columns[rfe.support_]

# Print selected features
print("Selected Features:", selected_features)

# Plotting feature importance
plt.figure(figsize=(12, 12))
plt.barh(range(len(rfe.support_)), rfe.support_)
plt.yticks(range(len(X.columns)), X.columns)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot (Wrapper Method - RFE)')
plt.show()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# Backward Selection using RFE
selector_backward = RFE(estimator=logreg, n_features_to_select=10, step=1)
X_backward_selected = selector_backward.fit_transform(X, y)
backward_selected_features = X.columns[selector_backward.support_]

# Forward Selection using RFE
selector_forward = RFE(estimator=logreg, n_features_to_select=10, step=1)
X_forward_selected = selector_forward.fit_transform(X, y)
forward_selected_features = X.columns[selector_forward.support_]

# Select K Best Features
selector_k_best = SelectKBest(score_func=f_classif, k=10)
X_k_best_selected = selector_k_best.fit_transform(X, y)
k_best_selected_features = X.columns[selector_k_best.get_support()]

# Plotting
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.barh(range(len(backward_selected_features)), logreg.coef_[0][selector_backward.support_])
plt.yticks(range(len(backward_selected_features)), backward_selected_features)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Backward Selection')

plt.subplot(1, 3, 2)
plt.barh(range(len(forward_selected_features)), logreg.coef_[0][selector_forward.support_])
plt.yticks(range(len(forward_selected_features)), forward_selected_features)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Forward Selection')

plt.subplot(1, 3, 3)
plt.barh(range(len(k_best_selected_features)), logreg.coef_[0][selector_k_best.get_support()])
plt.yticks(range(len(k_best_selected_features)), k_best_selected_features)
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Select K Best')

plt.tight_layout()
plt.show()

chi2_scores, _ = chi2(X, y)

# Calculate Fisher scores
fisher_scores, _ = f_classif(X, y)

# Get top features based on scores
top_chi2_features = [X.columns[i] for i in np.argsort(chi2_scores)[-10:]]
top_fisher_features = [X.columns[i] for i in np.argsort(fisher_scores)[-10:]]

print("Top 10 features based on chi-square:", top_chi2_features)
print("Top 10 features based on Fisher score:", top_fisher_features)

# Plotting chi-square feature importance 
plt.figure(figsize=(10, 6))
plt.barh(range(len(chi2_scores)), chi2_scores)
plt.yticks(range(len(X.columns)), X.columns)
plt.xlabel('Chi-Square Score')
plt.ylabel('Features')
plt.title('Feature Importance based on Chi-Square')
plt.show()

# Plotting Fisher score feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(fisher_scores)), fisher_scores)
plt.yticks(range(len(X.columns)), X.columns)
plt.xlabel('Fisher Score')
plt.ylabel('Features')
plt.title('Feature Importance based on Fisher Score')
plt.show()

# Make predictions on the test set
y_pred = logreg.predict(X_test)


cm = confusion_matrix(Y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
'''

# ---- Random-Forest ---- #
X=mushroom_data.drop('class', axis=1) 
y=mushroom_data['class'] 
X.head()
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])

Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=100)
print(f"Number of samples for training set:{X_train.shape}")
print(f"Number of samples for test set:{X_test.shape}")

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
print(f'\nGenauigkeit: {accuracy}')
print('\nKonfusionsmatrix:')
print(conf_matrix)
print('\nKlassifikationsbericht:')
print(class_report)

# Heatmap für die Korrelation der Merkmale
plt.figure(figsize=(12, 10))
sns.heatmap(mushroom_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korrelation der Merkmale')
plt.show()
