import plotly.express as px
from dataset import df_rec_por_dia

class GraficoVendas:
    def __init__(self, df_rec_mensal):
        self.df = df_rec_mensal

    def grafico_receita_mensal(self):
        # Cria o gráfico de receita mensal
        grafico_rec_mensal = px.bar(
            self.df,
            x='Mes',
            y='Preco_R$',
            color='Ano',
            title="Receita Mensal por Ano",
            labels={"Preco_R$": "Receita (R$)", "Mes": "Mês"},
            category_orders={"Mes": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]},
        )
        return grafico_rec_mensal

grafico_re_venda_por_dia = px.bar(
    df_rec_por_dia.head(7).reset_index(),
    x='Data',
    y='Preco_R$',
    text_auto=True,
    title='Top 7 de vendas por dia do mês',
    labels={'Data': 'Data da Venda', 'Preco_R$': 'Receita (R$)'}
)