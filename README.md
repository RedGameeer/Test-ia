# **Projeto Python Algodão - Dashboard de Vendas**

## **Descrição**

Este projeto tem como objetivo criar um dashboard interativo para análise de dados de vendas de algodão. Utilizando **Streamlit** para construção da interface web, **Pandas** para manipulação de dados e **Plotly** para gráficos interativos, a aplicação permite visualizar receitas, realizar processamento de dados e analisar a variação de preços ao longo do tempo. Também foi integrada uma **Rede Neural com TensorFlow/Keras** para realizar a **previsão do preço do algodão** com base em séries temporais. Agora o projeto também conta com **três modelos de previsão diferentes**: LSTM, MLP e Regressão Linear.

## **Pré-requisitos**

Antes de executar o projeto, é necessário configurar o ambiente de desenvolvimento local. Siga os passos abaixo para garantir que tudo esteja pronto.

## **1. Criação do Ambiente Virtual**

Primeiro, vamos criar e ativar o ambiente virtual para isolar as dependências do projeto.

### Passo 1.1 - Criando o Ambiente Virtual
Abra o terminal e execute o comando abaixo para criar o ambiente virtual:

```bash
python -m venv .venv
```

### Passo 1.2 - Habilitando o Ambiente Virtual
Ative o ambiente virtual para garantir que as dependências sejam instaladas corretamente:

- No Windows:
  ```bash
  .\.venv\Scripts\activate
  ```

- No macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

## **2. Instalando as Dependências**

Após ativar o ambiente virtual, você precisará instalar as bibliotecas necessárias para rodar o projeto. Use o comando abaixo:

### Passo 2.1 - Instalar via requirements.txt

```bash
pip install -r requirements.txt
```

### Conteúdo do `requirements.txt`

```txt
streamlit==1.34.0
plotly==5.20.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
tensorflow==2.16.1
```

### Passo 2.2 - Verificando as Dependências Instaladas

```bash
pip list
```

## **3. Executando o Projeto**

Agora que todas as dependências estão instaladas, você pode iniciar o servidor local do Streamlit para visualizar o dashboard interativo.

### Passo 3.1 - Executando o Streamlit

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador, geralmente acessível em `http://localhost:8501`.

## **4. Navegação no Dashboard**

- **Aba 1**: Exibe a visualização de vendas com a tabela de dados.
- **Aba 2**: Mostra o processamento de dados, com métricas como receita total e quantidade de vendas.
- **Aba 3**: Exibe gráficos interativos para análise dos preços do algodão ao longo do tempo.
- **Aba 4**: Traz a previsão do próximo preço do algodão com base em três modelos diferentes:
  - LSTM (Rede Neural Recorrente)
  - MLP (Rede Neural Densa)
  - Regressão Linear

---

### **Atualizando as Dependências do Projeto**

Se você adicionar novas dependências ao seu ambiente ou se outras pessoas precisarem instalar as mesmas dependências, basta atualizar o arquivo `requirements.txt` com o seguinte comando:

1. **Certifique-se de que o ambiente virtual esteja ativado**:
   - No Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - No macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

2. **Instalar novas dependências** com `pip`:
   ```bash
   pip install <nome-da-biblioteca>
   ```

3. **Atualizar o arquivo `requirements.txt`**:
   ```bash
   pip freeze > requirements.txt
   ```

---

### **Tecnologias Utilizadas**
- **Python**: Linguagem utilizada para o desenvolvimento do projeto.
- **Pandas**: Manipulação e análise de dados.
- **Plotly**: Visualização de gráficos interativos.
- **Streamlit**: Criação rápida de interfaces web interativas.
- **TensorFlow/Keras**: Biblioteca de aprendizado profundo usada para previsão do preço do algodão.
- **Scikit-learn**: Biblioteca de machine learning utilizada para Regressão Linear e outros modelos clássicos.
