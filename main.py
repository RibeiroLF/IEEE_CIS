import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


dados = pd.read_excel('C:\\Users\\desktop\\PycharmProjects\\IEEE_CIS\\Titanic.xlsx')
dados = dados.dropna(axis=1, how='all')

if 'Age' not in dados.columns or 'Survived' not in dados.columns:
    print("Os dados não foram filtrados corretamente. Certifique-se de que as colunas 'Age' e 'Survived' existam.")
    exit()

idades_filtradas = dados.dropna(subset=['Age']).copy()
imputer = SimpleImputer(strategy='mean')
dados_preenchidos = imputer.fit_transform(idades_filtradas[['Age']])

idades_filtradas.loc[:, 'Age_imputed'] = dados_preenchidos

X = idades_filtradas[['Age_imputed']]
y = idades_filtradas['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print("Matriz de Confusão: ")
print(confusion_mat)

plt.hist(idades_filtradas['Age_imputed'], bins=10, edgecolor='blue')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idades')
plt.show()

taxa_sobrevivencia = dados.dropna(subset=['Survived'])
labels = ['Sobreviveu', 'Não sobreviveu']
values = taxa_sobrevivencia['Survived'].value_counts()
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['blue', 'yellow'])

plt.title('Taxa de Sobrevivência')
plt.show()

X = idades_filtradas[['Age']]
y = idades_filtradas['Survived']

regressor = LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('Idade')
plt.ylabel('Sobrevivência')
plt.title('Regressão Linear: Idade x Sobrevivência')
plt.show()

irmaos_a_bordo = dados.dropna(subset=['SibSp'])
irmaos_a_bordo = irmaos_a_bordo[irmaos_a_bordo['SibSp'] != 0]

plt.hist(irmaos_a_bordo['SibSp'], bins=10, color='red', edgecolor='black')
plt.xlabel('Quantidade de irmãos/cônjuges a bordo')
plt.ylabel('Frequência')
plt.title('Quantidade de irmãos/cônjuges a bordo')

plt.show()

sexo_passageiro = dados.dropna(subset=['Sex'])
labels = ['Homem', 'Mulher']
values = sexo_passageiro['Sex'].value_counts()
plt.pie(values, labels=labels, autopct='%.1f%%', colors=['orange', 'green'])

plt.title('Sexo dos passageiros')
plt.show()

embarque_encoded = pd.get_dummies(dados['Embarked'], prefix='Embarked')
dados_encoded = pd.concat([dados, embarque_encoded], axis=1)

embarque = dados.dropna(subset=['Embarked'])
labels = ['S = Southampton','C = Cherbourg', 'Q = Queenstown']
values = embarque['Embarked'].value_counts()

plt.pie(values, labels=labels, autopct='%.1f%%', colors=['blue', 'white', 'grey'])
plt.show()

pais_a_bordo=dados.dropna(subset=['Parch'])
pais_a_bordo=pais_a_bordo[pais_a_bordo['Parch']!=0]
plt.hist(pais_a_bordo['Parch'], bins=15, color='orange')
plt.xlabel('Pais/Filhos a bordo')
plt.ylabel('Frequência')
plt.title('Pais/Filhos a bordo')

plt.show()

classe_passageiro = dados.dropna(subset=['Pclass'])
labels = ['3','1','2']
values=classe_passageiro['Pclass'].value_counts()
plt.pie(values, labels=labels,autopct='%.1f%%', colors= ['red', 'green', 'grey'])
plt.show()
