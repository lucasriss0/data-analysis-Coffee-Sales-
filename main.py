import os
import warnings
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)

# ==============================
# Importando a base de dados
# ==============================

try:
    path = kagglehub.dataset_download("ihelon/coffee-sales")
    df1 = pd.read_csv(f"{path}/index_1.csv")
except Exception:
    if os.path.exists("index_1.csv"):
        df1 = pd.read_csv("index_1.csv")
    else:
        raise FileNotFoundError(
            "Nao foi possivel baixar do Kaggle nem encontrar 'index_1.csv' na pasta atual."
        )

# ==============================
# Tratamento dos dados
# ==============================

df1['date'] = pd.to_datetime(df1['date'])
df1['datetime'] = pd.to_datetime(df1['datetime'])

df1['dia_semana'] = df1['date'].dt.day_name()
df1['hora'] = df1['datetime'].dt.hour

ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# ==============================
# Informações da base
# ==============================

print(df1.info())

print("\nEstatísticas Gerais")
print(df1.describe())

# ==============================
# Faturamento por café
# ==============================

df1.groupby('coffee_name')['money'].sum()\
    .sort_values()\
    .plot(kind='barh', color='saddlebrown')

plt.title('Faturamento por Café')
plt.xlabel('Faturamento (R$)')
plt.tight_layout()
plt.show()

# ==============================
# Faturamento total
# ==============================

faturamento_total = df1['money'].sum()
print(f"\nFaturamento Total: R$ {faturamento_total:.2f}")
print("Quantidade de vendas:", len(df1))

# ==============================
# Evolução diária (histórico)
# ==============================

# reindexa preenchendo dias sem nenhuma venda com 0 -- importante para a
# previsão mais adiante, senão dias "vazios" simplesmente somem do grupo
vendas_dia = df1.groupby('date')['money'].sum().asfreq('D').fillna(0)

plt.figure(figsize=(12, 5))
plt.plot(vendas_dia.index, vendas_dia.values, color='saddlebrown')
plt.title('Faturamento Diário')
plt.ylabel('Valor (R$)')
plt.tight_layout()
plt.show()

# ==============================
# Vendas por dia da semana
# ==============================

resultado = df1.groupby('dia_semana')['money'].sum().reindex(ordem)

resultado.plot(kind='bar', color='peru')
plt.title('Vendas por Dia da Semana')
plt.ylabel('Faturamento')
plt.tight_layout()
plt.show()

# ==============================
# Vendas por horário
# ==============================

vendas_hora = df1.groupby('hora')['money'].sum()

vendas_hora.plot(kind='bar', color='chocolate')
plt.title('Faturamento por Hora')
plt.xlabel('Hora')
plt.ylabel('Faturamento')
plt.tight_layout()
plt.show()

# ==============================
# Heatmap Hora x Dia
# ==============================

heatmap = df1.pivot_table(
    values='money',
    index='dia_semana',
    columns='hora',
    aggfunc='sum',
    fill_value=0
).reindex(ordem)

plt.figure(figsize=(13, 5))
sns.heatmap(heatmap, cmap='YlOrBr', cbar_kws={'label': 'Faturamento (R$)'})
plt.title("Faturamento por Hora x Dia da Semana")
plt.tight_layout()
plt.show()

# ==============================
# Estatísticas das vendas
# ==============================

print("\nResumo Estatístico")
print(df1['money'].describe())


# ==============================
# Clientes recorrentes
# ==============================

clientes = df1.dropna(subset=['card'])

resumo = clientes.groupby('card').agg(
    total_gasto=('money', 'sum'),
    compras=('money', 'count'),
    ticket_medio=('money', 'mean')
)

print("\nClientes identificados:", len(resumo))
print("Clientes recorrentes:", len(resumo[resumo['compras'] > 1]))

print("\nTop 10 clientes")
print(resumo.sort_values('total_gasto', ascending=False).head(10))

# ==============================
# Forma de pagamento
# ==============================

print("\nFaturamento por forma de pagamento")
print(df1.groupby('cash_type')['money'].agg(['count', 'sum', 'mean']))

df1.groupby('cash_type')['money'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel("")
plt.title("Faturamento por Forma de Pagamento")
plt.tight_layout()
plt.show()

# ==============================
# Correlação
# ==============================

diario = df1.groupby('date').agg(
    faturamento=('money', 'sum'),
    vendas=('money', 'count')
)

corr = diario['faturamento'].corr(diario['vendas'])
print(f"\nCorrelação entre número de vendas e faturamento: {corr:.2f}")

# ==============================
# Teste t (Semana x Final de Semana)
# ==============================

df1['fim_semana'] = df1['dia_semana'].isin(['Saturday', 'Sunday'])

grupo1 = df1[df1['fim_semana'] == False]['money']
grupo2 = df1[df1['fim_semana'] == True]['money']

t, p = stats.ttest_ind(grupo1, grupo2, equal_var=False)

print("\nTeste T")
print(f"p-valor: {p:.4f}")
if p < 0.05:
    print("Existe diferença significativa entre semana e final de semana.")
else:
    print("Não existe diferença significativa.")

# ==============================
# ANOVA
# ==============================

grupos = [grupo['money'] for _, grupo in df1.groupby('coffee_name')]
f, p = stats.f_oneway(*grupos)

print("\nANOVA")
print(f"p-valor: {p:.5f}")
if p < 0.05:
    print("Há diferença significativa entre os cafés.")
else:
    print("Não há diferença significativa.")

# ==============================
# Previsão utilizando Regressão Linear
# ==============================

# quantos dias no futuro prever -- ajuste aqui se quiser um horizonte diferente
DIAS_PREVISAO = 30

# quantos dias de histórico recente mostrar no gráfico de previsão
DIAS_HISTORICO_EXIBIDOS = 60

X = np.arange(len(vendas_dia)).reshape(-1, 1)
y = vendas_dia.values

modelo = LinearRegression()
modelo.fit(X, y)

# erro padrão dos resíduos, usado para desenhar uma banda de confiança
residuos = y - modelo.predict(X)
erro_padrao = residuos.std()

X_futuro = np.arange(len(vendas_dia), len(vendas_dia) + DIAS_PREVISAO).reshape(-1, 1)
previsao = modelo.predict(X_futuro)
previsao = np.clip(previsao, 0, None)

limite_superior = previsao + 1.96 * erro_padrao
limite_inferior = np.clip(previsao - 1.96 * erro_padrao, 0, None)

datas_futuras = pd.date_range(vendas_dia.index[-1] + pd.Timedelta(days=1), periods=DIAS_PREVISAO)

# ---- Gráfico 1: apenas o histórico completo ----
plt.figure(figsize=(12, 5))
plt.plot(vendas_dia.index, y, color='saddlebrown')
plt.title("Faturamento Diário - Histórico Completo")
plt.ylabel("Faturamento (R$)")
plt.tight_layout()
plt.show()

# ---- Gráfico 2: previsão isolada, com horizonte maior e banda de confiança ----
plt.figure(figsize=(13, 6))
plt.plot(
    vendas_dia.index[-DIAS_HISTORICO_EXIBIDOS:],
    y[-DIAS_HISTORICO_EXIBIDOS:],
    label='Histórico recente',
    color='gray',
    alpha=0.6
)
plt.plot(datas_futuras, previsao, '--', color='crimson', linewidth=2, label=f'Previsão ({DIAS_PREVISAO} dias)')
plt.fill_between(
    datas_futuras, limite_inferior, limite_superior,
    color='crimson', alpha=0.15, label='Intervalo de confiança (95%)'
)
plt.axvline(vendas_dia.index[-1], color='black', linestyle=':', alpha=0.5)
plt.title(f"Previsão de Faturamento - Próximos {DIAS_PREVISAO} dias")
plt.ylabel("Faturamento (R$)")
plt.legend()
plt.tight_layout()
plt.show()

tendencia_dia = modelo.coef_[0]
direcao = "crescimento" if tendencia_dia > 0 else "queda"

print(f"\nTendência: {direcao} de aproximadamente R$ {abs(tendencia_dia):.2f} por dia")
print(f"\nFaturamento previsto para os próximos {DIAS_PREVISAO} dias:")
print(f"R$ {previsao.sum():.2f}")
print(f"Média diária prevista: R$ {previsao.mean():.2f}")

print("\nAnálise concluída!")