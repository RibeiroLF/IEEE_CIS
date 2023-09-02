import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Carregar apenas os primeiros 500 dados
data = pd.read_csv("creditcard.csv", nrows=500)

#Selecionar apenas as colunas V1 a V28 e a label
selected_columns = ['V' + str(i) for i in range(1, 29)]
selected_columns.append('Class')
data = data[selected_columns]

#Separar a label das features
X = data.drop("Class", axis=1).values
y = data["Class"].values

#Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Redimensionar os dados para o intervalo entre -1 e 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Derivada da função de ativação sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

#Parâmetros da rede neural e Regularização L2
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.1
lambda_reg = 0.001
epochs = 1000

#Inicialização dos pesos com inicialização de Xavier
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))

#Treinamento do perceptron com regularização L2
for epoch in range(epochs):
    hidden_layer_input = np.dot(X_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    error = y_train.reshape(-1, 1) - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Atualização dos pesos com regularização
    weights_hidden_output += hidden_layer_output.T.dot(
        d_predicted_output) * learning_rate - lambda_reg * weights_hidden_output
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate - lambda_reg * weights_input_hidden

#Previsões no conjunto de teste
hidden_layer_input = np.dot(X_test, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)
predicted_labels = (predicted_output > 0.5).astype(int)

#Avaliar o modelo
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

#Matriz de confusão e relatório de classificação
confusion = confusion_matrix(y_test, predicted_labels)
print("Matriz de Confusão:")
print(confusion)

classification_rep = classification_report(y_test, predicted_labels)
print("\nRelatório de Classificação:")
print(classification_rep)