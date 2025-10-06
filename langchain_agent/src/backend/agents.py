"""
Agente Orquestrador e Agente Especialista (como Ferramenta)

Este módulo define uma arquitetura de dois agentes:
1.  **Agente Orquestrador**: O ponto de entrada principal. Ele gerencia o fluxo da conversa,
    decide qual ferramenta usar e delega tarefas complexas de análise de dados.
2.  **Agente Especialista**: Utilizado como uma ferramenta pelo orquestrador. É focado
    exclusivamente em executar tarefas de análise sobre o DataFrame pandas, usando
    suas próprias ferramentas de dados.
"""

import io
import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from plotly.basedatatypes import BaseFigure
from pytz import timezone

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Ferramentas do Agente ESPECIALISTA ---


@tool
def get_dataset_summary() -> str:
    """
    Retorna um resumo completo do DataFrame, incluindo informações sobre os tipos de dados,
    valores não nulos e estatísticas descritivas. Ideal para uma visão geral inicial.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    summary_str = df.describe(include="all").to_string()
    return f"Resumo das Informações:\n{info_str}\n\nEstatísticas Descritivas:\n{summary_str}"


@tool
def list_columns() -> str:
    """Lista todas as colunas disponíveis no DataFrame."""
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    return f"As colunas disponíveis são: {', '.join(df.columns.tolist())}"


@tool
def count_rows_and_columns() -> str:
    """Conta o número de linhas e colunas no DataFrame."""
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    return f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas."


@tool
def get_value_counts(column: str) -> str:
    """
    Retorna a contagem de valores únicos para uma coluna categórica.
    Útil para entender a distribuição e frequência de categorias.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if column not in df.columns:
        return f"Erro: A coluna '{column}' não existe no dataset."
    if df[column].dtype not in ["object", "category"]:
        return f"Erro: Esta função é mais adequada para colunas categóricas. A coluna '{column}' é numérica."
    return df[column].value_counts().to_string()


@tool
def calculate_correlation_matrix() -> str:
    """
    Calcula e retorna a matriz de correlação para as colunas numéricas do DataFrame.
    Essencial para entender as relações lineares entre as variáveis.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return "Não há colunas numéricas suficientes para calcular a correlação."
    return numeric_df.corr().to_string()


@tool
def handle_missing_values() -> str:
    """
    Verifica e retorna um resumo dos valores ausentes (NaN) em cada coluna do DataFrame.
    Essencial para entender a qualidade e a integridade dos dados.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Contagem": missing_values, "Porcentagem (%)": missing_percentage}
    )
    missing_df = missing_df[missing_df["Contagem"] > 0]
    if missing_df.empty:
        return "Não foram encontrados valores ausentes no dataset."
    return f"Valores ausentes encontrados:\n{missing_df.to_string()}"


@tool
def detect_outliers_iqr(column: str) -> str:
    """
    Detecta outliers em uma coluna numérica usando o método do Intervalo Interquartil (IQR).
    Retorna um resumo dos outliers encontrados.
    """
    df = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if column not in df.columns:
        return f"Erro: A coluna '{column}' não existe no dataset."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Erro: A detecção de outliers só pode ser feita em colunas numéricas. A coluna '{column}' não é numérica."

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    if outliers.empty:
        return f"Nenhum outlier detectado na coluna '{column}' usando o método IQR."
    else:
        outliers_summary = outliers[column].describe().to_string()
        return (
            f"Detectados {len(outliers)} outliers na coluna '{column}'.\n"
            f"- Limite inferior: {lower_bound:.2f}\n"
            f"- Limite superior: {upper_bound:.2f}\n\n"
            f"Resumo estatístico dos outliers:\n{outliers_summary}"
        )


@tool
def generate_histogram(column: str) -> str:
    """
    Gera um histograma para uma coluna numérica e o armazena para exibição.
    Use esta ferramenta quando o usuário pedir um 'histograma', 'distribuição' ou 'gráfico' de uma variável.
    O gráfico será exibido na interface do usuário. Retorne uma confirmação para o usuário.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if column not in df.columns or not pd.api.types.is_numeric_dtype(df.get(column)):
        return f"Erro: A coluna '{column}' não é numérica ou não existe."

    description = df[column].describe().to_string()
    metadata = f"Metadados da distribuição de '{column}':\n{description}"

    fig = px.histogram(df, x=column, title=f"Distribuição de {column}")
    st.session_state.charts.append(fig)
    return f"O histograma da coluna '{column}' foi gerado e será exibido.\n\n{metadata}"


@tool
def generate_bar_chart(column: str) -> str:
    """
    Gera um gráfico de barras para uma coluna categórica e o armazena para exibição.
    Use para visualizar a frequência de categorias, quando o usuário pedir um 'gráfico de barras'.
    O gráfico será exibido na interface do usuário. Retorne uma confirmação para o usuário.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if column not in df.columns:
        return f"Erro: A coluna '{column}' não existe no dataset."

    # Permite colunas numéricas com baixa cardinalidade (poucos valores únicos)
    is_categorical = df[column].dtype in ["object", "category"]
    is_low_cardinality_numeric = (
        pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() <= 20
    )

    if not is_categorical and not is_low_cardinality_numeric:
        return f"Erro: A coluna '{column}' não é adequada para um gráfico de barras. Se for numérica com muitos valores, tente um histograma."

    counts = df[column].value_counts().nlargest(20)
    metadata = f"Metadados das 20 categorias mais frequentes de '{column}':\n{counts.to_string()}"

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title=f"Contagem de {column}",
        labels={"x": column, "y": "Contagem"},
    )
    st.session_state.charts.append(fig)
    return f"O gráfico de barras da coluna '{column}' foi gerado e será exibido.\n\n{metadata}"


@tool
def generate_scatter_plot(x_column: str, y_column: str) -> str:
    """
    Gera um gráfico de dispersão (scatter plot) para visualizar a relação entre duas colunas numéricas.
    Use quando o usuário pedir para 'relacionar', 'comparar' ou 'plotar' duas variáveis.
    O gráfico será exibido na interface do usuário. Retorne uma confirmação para o usuário.
    """
    df = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if x_column not in df.columns or not pd.api.types.is_numeric_dtype(
        df.get(x_column)
    ):
        return f"Erro: A coluna X '{x_column}' não é numérica ou não existe."
    if y_column not in df.columns or not pd.api.types.is_numeric_dtype(
        df.get(y_column)
    ):
        return f"Erro: A coluna Y '{y_column}' não é numérica ou não existe."

    sample_data = df[[x_column, y_column]].sample(min(5, len(df))).to_string()
    metadata = f"Amostra de dados para '{x_column}' vs '{y_column}':\n{sample_data}"

    fig = px.scatter(
        df, x=x_column, y=y_column, title=f"Relação entre {x_column} e {y_column}"
    )
    st.session_state.charts.append(fig)
    return f"O gráfico de dispersão entre '{x_column}' e '{y_column}' foi gerado e será exibido.\n\n{metadata}"


@tool
def generate_box_plot(column: str) -> str:
    """
    Gera um box plot para uma coluna numérica, útil para visualizar distribuição e outliers.
    Use quando o usuário pedir um 'box plot' ou quiser ver 'outliers' e 'distribuição' de forma gráfica.
    """
    df = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if column not in df.columns or not pd.api.types.is_numeric_dtype(df.get(column)):
        return f"Erro: A coluna '{column}' não é numérica ou não existe."

    description = df[column].describe().to_string()
    metadata = f"Metadados do box plot de '{column}':\n{description}"

    fig = px.box(df, y=column, title=f"Box Plot de {column}")
    st.session_state.charts.append(fig)
    return f"O box plot da coluna '{column}' foi gerado e será exibido.\n\n{metadata}"


@tool
def retrieve_dataframe_rows(n_rows: int = 5, part: str = "head") -> str:
    """
    Recupera e exibe as primeiras (head), últimas (tail) ou uma amostra aleatória (sample) de linhas do DataFrame.
    Use esta ferramenta quando o usuário pedir para 'ver os dados', 'mostrar algumas linhas', 'ver o início/fim dos dados'.

    Args:
        n_rows (int): O número de linhas a serem recuperadas. O padrão é 5.
        part (str): A parte do DataFrame a ser recuperada. Pode ser 'head', 'tail' ou 'sample'. O padrão é 'head'.
    """
    df: pd.DataFrame = st.session_state.get("current_dataframe")

    if df is None or df.empty:
        return "Não foi possível continuar com a análise, o DataFrame está vazio."

    if part not in ["head", "tail", "sample"]:
        return f"Erro: O método '{part}' é inválido. Use 'head', 'tail' ou 'sample'."

    if not isinstance(n_rows, int) or n_rows <= 0:
        n_rows = 5  # Default to 5 if invalid number is passed

    try:
        if part == "head":
            data = df.head(n_rows)
        elif part == "tail":
            data = df.tail(n_rows)
        else:  # sample
            data = df.sample(min(n_rows, len(df)))  # Avoid error if n_rows > len(df)

        return f"Exibindo {len(data)} linhas (método: {part}):\n{data.to_string()}"
    except Exception as e:
        return f"Ocorreu um erro ao recuperar as linhas: {e}"


@tool
def generate_custom_code(code: str) -> str:
    """
    Executa código Python para análises personalizadas ou geração de gráficos complexos. O ambiente de execução tem acesso ao 'plotly.express' como 'px' e a funções especiais.

    Funções disponíveis no ambiente:
    - `get_dataframe()`: Retorna o DataFrame pandas atual para análise. Use-o como `df = get_dataframe()`.
    - `save_chart(fig)`: Salva uma figura Plotly para exibição. O código deve gerar a figura e passá-la para esta função.

    **Para GERAR GRÁFICOS:**
    1. Obtenha o dataframe: `df = get_dataframe()`
    2. Crie a figura: `fig = px.scatter_3d(df, ...)`
    3. Salve o gráfico: `save_chart(fig)`

    **Para CÁLCULOS ou manipulação de dados:**
    O resultado da última linha de código será retornado.
    Exemplo:
    `df = get_dataframe()`
    `df[df['coluna'] > 100].describe()`

    Use esta ferramenta para solicitações que as outras não atendem, como gráficos 3D, heatmaps, ou análises estatísticas personalizadas.
    """

    def get_dataframe() -> pd.DataFrame:
        """Função auxiliar para obter o DataFrame do estado da sessão."""
        return st.session_state.get("current_dataframe")

    def save_chart(fig: BaseFigure):
        """Função auxiliar para salvar o gráfico no estado da sessão."""
        st.session_state.charts.append(fig)

    # Cria um ambiente REPL seguro com as funções auxiliares
    python_repl = PythonAstREPLTool(
        globals=globals(),
        locals={
            "px": px,
            "go": go,
            "get_dataframe": get_dataframe,
            "save_chart": save_chart,
        },
    )
    try:
        result = python_repl.run(code)
        return f"Código executado com sucesso. Resultado:\n{result}"
    except Exception as e:
        return f"Erro ao executar o código Python: {e}"


# --- Lógica do Agente ---


def get_llm(provider: str, model_name: str) -> Any:
    """Fábrica para criar instâncias de LLM com base no provedor."""
    if provider == "Google":
        api_key = st.secrets["google_ai"].get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Chave de API do Google não encontrada em secrets.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    elif provider == "Groq":
        api_key = st.secrets["groq_ai"].get("GROQ_API_KEY")

        if not api_key:
            raise ValueError("Chave de API do Groq não encontrada em secrets.")

        return ChatGroq(model_name=model_name, groq_api_key=api_key)
    else:
        raise ValueError(f"Provedor de LLM desconhecido: {provider}")


SPECIALIST_SYSTEM_PROMPT = """
Você é um assistente especialista em análise de dados.
Seu objetivo é ajudar os usuários a entenderem seus dados contidos em um DataFrame pandas.
Responda às perguntas do usuário da melhor forma possível, utilizando as ferramentas disponíveis.

**Diretrizes:**
- O DataFrame está disponível implicitamente para as ferramentas. Você não precisa passá-lo como argumento.
- Existem ferramentas para dados classificados como categóricos (não numéricos), verifique se o dado é o correto antes de usar a ferramenta.
- Não invente informações, responda o que for possível com seu conhecimento + ferramentas disponíveis.
- Para perguntas sobre visualizações gráficas, escolha a ferramenta mais apropriada (histograma, gráfico de barras, etc.) com base nos valores.
- As ferramentas de geração de gráficos retornam uma confirmação e um campo de 'Metadados' com um resumo dos dados. Use esses metadados para interpretar o gráfico e fornecer uma explicação rica e com insights para o usuário.
- Para solicitações complexas, cálculos personalizados ou gráficos que as ferramentas padrão não suportam (ex: 3D, heatmaps), use a ferramenta `generate_custom_code`.
- Ao usar `generate_custom_code` para gráficos, sempre use a função `save_chart(fig)` para salvar a visualização. Para obter os dados, use `df = get_dataframe()`.
- Se o usuário pedir um gráfico sem especificar a coluna, você **DEVE** perguntar qual coluna ele deseja analisar, desde que o gráfico escolhido seja específico.
- Para perguntas sobre correlação ou relação entre múltiplas variáveis, comece com `calculate_correlation_matrix` para análise textual.
- Após utilizar uma ferramenta, crie uma resposta objetiva e com insights, se possível, para explicar os dados ou a visualização.
"""


class SpecialistAgent:
    """Agente especialista em análise de dados."""

    def __init__(self):
        self.df = st.session_state.get("current_dataframe")
        if self.df is None:
            raise ValueError(
                "DataFrame não encontrado no st.session_state para o SpecialistAgent."
            )
        # O Especialista usa um modelo potente e específico para análise.
        self.llm = get_llm(
            provider=st.session_state.get("selected_provider"),
            model_name=st.session_state.get("selected_model"),
        )

        self.tools = [
            get_dataset_summary,
            list_columns,
            count_rows_and_columns,
            get_value_counts,
            calculate_correlation_matrix,
            handle_missing_values,
            detect_outliers_iqr,
            generate_histogram,
            generate_bar_chart,
            generate_scatter_plot,
            generate_box_plot,
            retrieve_dataframe_rows,
            generate_custom_code,
        ]

        # O prompt permanece o mesmo, pois as ferramentas são auto-descritivas
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SPECIALIST_SYSTEM_PROMPT),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=7,
        )

    def run(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Executa uma consulta no agente."""
        return self.agent_executor.invoke({"input": query})


# --- Lógica do Agente Orquestrador ---


@tool("DataAnalyzer")
def run_data_analyzer(query: str) -> str:
    """
    Use esta ferramenta para qualquer pergunta que envolva análise de dados, estatísticas,
    contagem, gráficos, ou informações sobre o dataset. O input deve ser uma string da pergunta
    completa do usuário sobre os dados.
    """

    if not query:
        return "Erro: a consulta para o analisador de dados está vazia."

    specialist_agent = SpecialistAgent()

    response = specialist_agent.run(query)
    return response.get("output", "Não foi possível processar a sua solicitação.")


@tool("Get_date")
def get_current_date():
    """Função que retorna a data e hora no Brasil."""
    return datetime.now(timezone("America/Sao_Paulo")).strftime("%d/%m/%Y %H%M%S")


ORCHESTRATOR_SYSTEM_PROMPT = """
Você é um agente orquestrador. Sua principal função é entender a intenção do usuário e delegar a tarefa para a ferramenta correta.

**Diretrizes de Delegação:**
- Para qualquer tarefa que envolva análise de dados, estatísticas, contagem, gráficos, ou informações sobre o dataset, você **DEVE** usar a ferramenta `run_data_analyzer`.
- Para conversas gerais que não requerem análise de dados, responda diretamente.
- Você tem disponível uma ferramenta que retorna a data e hora, use quando necessário.

**Outras Regras:**
- Não invente informações, se não for possível informe ao usuário.
- Utilize emojis na conversa.
- Adicione sugestões de perguntas ou detalhes para o usuário, mantendo a conversa interativa.
- Para as respostas retornadas pelo agente de análise de dados, retorne ao usuário como resposta final se assertiva com o input do usuário.
"""


class OrchestratorAgent:
    """Agente orquestrador que gerencia o fluxo e delega para ferramentas, incluindo o agente especialista."""

    def __init__(self):
        # Busca a configuração do LLM e o DataFrame do estado da sessão do Streamlit
        model_name = st.session_state.get("selected_model")
        provider = st.session_state.get("selected_provider")

        if not model_name or not provider:
            raise ValueError(
                "Configuração de modelo/provedor não encontrada no st.session_state."
            )

        llm = get_llm(provider, model_name)
        self.llm = llm
        self.dataframe = st.session_state.get("current_dataframe")
        if self.dataframe is None:
            raise ValueError(
                "DataFrame não encontrado no st.session_state para o OrchestratorAgent."
            )

        # A memória agora é injetada a partir do st.session_state
        self.memory = st.session_state.get("orchestrator_memory")
        if self.memory is None:
            raise ValueError(
                "Objeto de memória não encontrado no st.session_state para o OrchestratorAgent."
            )

        self.tools = [run_data_analyzer, get_current_date]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ORCHESTRATOR_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent = AgentExecutor(
            agent=agent,
            memory=self.memory,
            tools=self.tools,
            max_iterations=7,
            verbose=True,
            handle_parsing_errors=True,
        )

    def process_user_query(self, user_query: str) -> Dict[str, Any]:
        """Processa a consulta do usuário através do agente orquestrador."""
        logger.info(f"Processando consulta: {user_query}")

        return self.agent.invoke({"input": user_query})
