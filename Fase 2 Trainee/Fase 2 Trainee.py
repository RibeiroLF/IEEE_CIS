import pandas as pd
vinho = pd.read_csv('winequality.csv')
#Tratamento de valores vazios
vinho = vinho.dropna()
#Expansão das colunas
vinho_separado = vinho.applymap(lambda x: x.split(',') if isinstance(x, str) else x)
print(vinho_separado)