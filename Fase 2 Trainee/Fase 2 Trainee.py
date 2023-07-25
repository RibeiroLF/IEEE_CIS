import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
vinho = pd.read_csv('winequality.csv')
#Tratamento de valores vazios
vinho = vinho.dropna()
#Expansão das colunas
vinho_separado = vinho.applymap(lambda x: x.split(',') if isinstance(x, str) else x)
print(vinho_separado)
X = vinho.drop('wine_is_red', axis=1)
y = vinho['wine_is_red']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=5, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisão do modelo:", accuracy)
