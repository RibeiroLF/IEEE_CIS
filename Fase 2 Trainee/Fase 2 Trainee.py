import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

vinho = pd.read_csv('winequality.csv')

#Tratamento de valores vazios
vinho = vinho.dropna()

#Convertendo para binário (vinho tinto ou não)
vinho['wine_is_red'] = (vinho['wine_is_red'] == 'red').astype(int)

#Dividindo característica e variável
X = vinho.drop('quality', axis=1).values
y_red = vinho['wine_is_red'].values
y_quality = vinho['quality'].values

#Padronizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Dividindo em teste e treino
X_train, X_test, y_train_red, y_test_red, y_train_quality, y_test_quality = train_test_split(
    X, y_red, y_quality, test_size=0.2, random_state=42
)

#Definindo o KNN usando NumPY
def previsão_knn(X_train, X_test, y_train, k=3):
    y_pred = []
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        y_pred.append(np.bincount(k_nearest_labels).argmax())
    return np.array(y_pred)

#KNN para previsão binária
y_prev_knn_binario = previsão_knn(X_train, X_test, y_train_red, k=5)
precisao_knn_binario = accuracy_score(y_test_red, y_prev_knn_binario)
print(f'Precisão da classificação binária (vinho tinto ou não): {precisao_knn_binario:.2f}')

#KNN para previsão multiclasse
def previsao_knn_multi(X_train, X_test, y_train, k=3):
    y_pred = []
    for x_test in X_test:
        distances = np.linalg.norm(X_train - x_test, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        y_pred.append(np.bincount(k_nearest_labels, minlength=11).argmax())  # Assuming wine quality ranges from 0 to 10
    return np.array(y_pred)

y_pred_knn_multiclass = previsao_knn_multi(X_train, X_test, y_train_quality, k=5)
knn_precisao_multi = accuracy_score(y_test_quality, y_pred_knn_multiclass)
print(f'Precisão da classificação multiclasse (prever qualidade do vinho): {knn_precisao_multi:.2f}')

#Modelo KNN para previsão
y_pred_knn = previsão_knn(X_train, X_test, y_train_quality, k=5)
resultado_precisao_knn = accuracy_score(y_test_quality, y_pred_knn)
print(f'Precisão do modelo KNN: {resultado_precisao_knn:.2f}')

#Random Forest no conjunto de dados original
rf_model_original = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_original.fit(X_train, y_train_quality)

#Precisão do Random Forest no conjunto de dados original
y_pred_rf_original = rf_model_original.predict(X_test)
precisao_rf_original = accuracy_score(y_test_quality, y_pred_rf_original)
print(f'Precisão do modelo Random Forest no conjunto de dados original: {precisao_rf_original:.2f}')

#Undersampling e Oversampling nos dados de classificação multiclasse
rus = RandomUnderSampler(random_state=42)
ros = RandomOverSampler(random_state=42)

X_train_under, y_train_quality_under = rus.fit_resample(X_train, y_train_quality)
X_train_over, y_train_quality_over = ros.fit_resample(X_train, y_train_quality)

#KNN para classificação multiclasse com Undersampling
y_pred_knn_multiclass_under = previsao_knn_multi(X_train_under, X_test, y_train_quality_under, k=5)
precisao_knn_multiclass_under = accuracy_score(y_test_quality, y_pred_knn_multiclass_under)
print(f'Precisão da classificação multiclasse com Undersampling: {precisao_knn_multiclass_under:.2f}')

#KNN para classificação multiclasse com Oversampling
y_pred_knn_multiclass_over = previsao_knn_multi(X_train_over, X_test, y_train_quality_over, k=5)
precisao_knn_multiclass_over = accuracy_score(y_test_quality, y_pred_knn_multiclass_over)
print(f'Precisão da classificação multiclasse com Oversampling: {precisao_knn_multiclass_over:.2f}')

#Random Forest com Oversampling
rf_model_over = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_over.fit(X_train_over, y_train_quality_over)
y_pred_rf_over = rf_model_over.predict(X_test)
rf_over_precisao = accuracy_score(y_test_quality, y_pred_rf_over)
print(f'Precisão do Random Forest com OverSampling: {rf_over_precisao:.2f}')

#Importâncias de cada atributo
importancia = rf_model_over.feature_importances_
indices = np.argsort(importancia)[::-1]

print("Importância de cada atributo:")
for i in indices:
    print(f"{vinho.columns[i]}: {importancia[i]:.2f}")
