import json
import pandas as pd

# Abre o arquivo JSON e carrega os dados
with open('dados/dados_cepea_novo.json', encoding='utf-8') as file:
    data = json.load(file)

# Cria o DataFrame
df = pd.DataFrame.from_dict(data)

# Ajusta as colunas
df.columns = ['Data', 'Preco_R$', 'Preco_US$']

# Filtra só linhas onde 'Data' é realmente uma data (dia/mês/ano)
df = df[df['Data'].str.match(r'\d{2}/\d{2}/\d{4}', na=False)]

# Converte para datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# Agrupamento mensal para receita
df_rec_mensal = df.set_index('Data').groupby(pd.Grouper(freq='MS'))['Preco_R$'].sum().reset_index()

# Adiciona Ano e Mês
df_rec_mensal['Ano'] = df_rec_mensal['Data'].dt.year
df_rec_mensal['Mes'] = df_rec_mensal['Data'].dt.month_name()

# Dataframe de receita por mes
df_rec_por_dia = df.groupby('Data')[['Preco_R$']].sum().sort_values('Preco_R$', ascending=False)

