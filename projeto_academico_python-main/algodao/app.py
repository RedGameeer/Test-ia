import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dataset import df, df_rec_mensal
from utils import format_number
from graficos import GraficoVendas, grafico_re_venda_por_dia
from previsoes import (
    previsao_lstm, previsao_mlp, previsao_reg_linear,
    rmse_lstm_reais, rmse_mlp_reais, rmse_lr_reais
)

# Configuração inicial da página
st.set_page_config(
    page_title="Dashboard de Vendas",
    page_icon=":shopping_trolley:",
    layout="wide"
)

# Título principal
st.title("Dashboard de Análise de Vendas :shopping_trolley:")

# Criando as abas
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📊 Visualização de Dados",
    "📈 Análise de Receita",
    "💰 Análise de Preços",
    "🤖 Previsões com IA",
    "📉 Comparação de Modelos"
])

# Aba 1 - Visualização de dados
with aba1:
    st.header("Dados Completos de Vendas")
    st.dataframe(df.sort_values('Data', ascending=False), height=500)
    
    with st.expander("🔍 Sobre os dados"):
        st.markdown("""
        - **Data**: Data da venda
        - **Preco_R$**: Preço do algodão em Reais
        - **Quantidade**: Quantidade vendida
        """)

# Aba 2 - Análise de receita
with aba2:
    st.header("Análise de Receita Mensal")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Receita Total", 
            value=format_number(df['Preco_R$'].sum(), 'R$'))
    with col2:
        st.metric(
            label="Total de Vendas", 
            value=format_number(df.shape[0]))
    
    # Gráfico de receita mensal
    grafico_vendas = GraficoVendas(df_rec_mensal)
    fig_receita = grafico_vendas.grafico_receita_mensal()
    st.plotly_chart(fig_receita, use_container_width=True)

# Aba 3 - Análise de preços
with aba3:
    st.header("Análise Histórica de Preços")
    
    preco_max = df.loc[df['Preco_R$'].idxmax()]
    preco_min = df.loc[df['Preco_R$'].idxmin()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Preço Máximo Histórico", 
            value=format_number(preco_max['Preco_R$'], 'R$'),
            delta=preco_max['Data'].strftime('%d/%m/%Y'))
    with col2:
        st.metric(
            label="Preço Mínimo Histórico", 
            value=format_number(preco_min['Preco_R$'], 'R$'),
            delta=preco_min['Data'].strftime('%d/%m/%Y'))
    
    # Gráfico de evolução de preços
    fig_precos = px.line(
        df, 
        x='Data', 
        y='Preco_R$',
        title="Evolução do Preço do Algodão",
        labels={'Preco_R$': 'Preço (R$)', 'Data': 'Data'}
    )
    st.plotly_chart(fig_precos, use_container_width=True)
    
    # Gráfico adicional de vendas por dia
    st.plotly_chart(grafico_re_venda_por_dia, use_container_width=True)

# Aba 4 - Previsões com IA
with aba4:
    st.header("Previsões de Preço com Inteligência Artificial")
    
    st.info("""
    Os modelos abaixo foram treinados para prever o próximo preço do algodão
    com base nos dados históricos. O RMSE indica o erro médio em Reais.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Rede Neural LSTM")
        previsao = previsao_lstm()
        st.metric(
            label="Próxima Previsão", 
            value=format_number(previsao, 'R$'))
        st.metric(
            label="Erro Médio (RMSE)", 
            value=format_number(rmse_lstm_reais, 'R$'))
        
    with col2:
        st.subheader("Rede Neural MLP")
        previsao = previsao_mlp()
        st.metric(
            label="Próxima Previsão", 
            value=format_number(previsao, 'R$'))
        st.metric(
            label="Erro Médio (RMSE)", 
            value=format_number(rmse_mlp_reais, 'R$'))
        
    with col3:
        st.subheader("Regressão Linear")
        previsao = previsao_reg_linear()
        st.metric(
            label="Próxima Previsão", 
            value=format_number(previsao, 'R$'))
        st.metric(
            label="Erro Médio (RMSE)", 
            value=format_number(rmse_lr_reais, 'R$'))
    
    with st.expander("📚 Sobre os modelos"):
        st.markdown("""
        - **LSTM**: Rede neural especializada em séries temporais
        - **MLP**: Rede neural tradicional para problemas genéricos
        - **Regressão Linear**: Modelo estatístico básico como referência
        """)

# Aba 5 - Comparação de modelos
with aba5:
    st.header("Comparação de Desempenho dos Modelos")
    
    # Dados para o gráfico
    modelos = ['LSTM', 'MLP', 'Regressão Linear']
    erros = [rmse_lstm_reais, rmse_mlp_reais, rmse_lr_reais]
    
    # Criando o gráfico de barras
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=modelos,
        y=erros,
        text=[f'R${erro:.2f}' for erro in erros],
        textposition='auto',
        marker_color=['#636EFA', '#EF553B', '#00CC96']
    ))
    
    fig.update_layout(
        title='Erro Médio (RMSE) em Reais',
        xaxis_title='Modelos',
        yaxis_title='Erro Médio (R$)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise comparativa
    melhor_modelo = modelos[np.argmin(erros)]
    st.success(f"✅ O modelo com melhor desempenho foi: **{melhor_modelo}**")
    
    with st.expander("📝 Interpretação dos resultados"):
        st.markdown("""
        - **RMSE (Root Mean Squared Error)**: 
          Indica o erro médio das previsões em Reais.
          Quanto menor, melhor o modelo.
        - Comparação direta entre abordagens diferentes.
        - Valores em R$ permitem análise prática do impacto.
        """)

# Rodapé
st.divider()
st.caption("Dashboard desenvolvido para análise de vendas - © 2025")