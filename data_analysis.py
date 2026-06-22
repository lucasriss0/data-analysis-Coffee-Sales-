#                Import das bibliotecas

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt

#                import da tabela
path = kagglehub.dataset_download("ihelon/coffee-sales")

#                Verifique quais arquivos existem
#for arquivo in os.listdir(path):
#print(arquivo)

#                    Salvando as planilhas em variaveis
df1 = pd.read_csv(f"{path}/index_1.csv")  
df2 = pd.read_csv(f"{path}/index_2.csv")  



#                   verificar os dados 

# df1.info()
# df1.describe()
# print(df1.columns)
# print(df2.columns)

#                primeiro Grafico contendo as vendas por categoria do café.
df1.groupby('coffee_name')['money'].sum()\
    .sort_values()\
    .plot(kind='barh')

plt.title('Faturamento por Café')
plt.show()


#                       faturamento
faturamento_total = df1['money'].sum()
print(f"Faturamento Total: R$ {faturamento_total:.2f}")

#                      Quantidades de vendas
print("Quantidade de vendas:", len(df1))



df1['date'] = pd.to_datetime(df1['date'])
vendas_dia = df1.groupby('date')['money'].sum()
vendas_dia.plot(figsize=(12,5))
plt.title('Faturamento Diário')
plt.ylabel('Valor')
plt.show()



df1['dia_semana'] = df1['date'].dt.day_name()
resultado = df1.groupby('dia_semana')['money'].sum()
ordem = [
    'Monday','Tuesday','Wednesday',
    'Thursday','Friday','Saturday','Sunday'
]
resultado = resultado.reindex(ordem)
resultado.plot(kind='bar')
plt.title('Vendas por Dia da Semana')
plt.show()