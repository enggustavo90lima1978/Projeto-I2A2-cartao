"""
Página Web Completa - Streamlit com Backend Integrado
Sistema completo de análise de dados com agentes inteligentes
"""

import zipfile
from typing import Dict, List

import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

# Imports do backend
from backend.agents import OrchestratorAgent

# --- Funções de Processamento de Dados ---


def _load_csv(file_like_object, **kwargs) -> pd.DataFrame | None:
    """Lê um objeto semelhante a um arquivo CSV e retorna um DataFrame."""
    try:
        return pd.read_csv(file_like_object, **kwargs)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")
        return None


def _load_from_zip(uploaded_file, **kwargs) -> pd.DataFrame | None:
    """Extrai e lê um arquivo CSV de um arquivo ZIP."""
    try:
        with zipfile.ZipFile(uploaded_file) as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                st.error("Nenhum arquivo .csv encontrado no arquivo .zip.")
                return None
            # Carrega o primeiro CSV encontrado no ZIP
            with z.open(csv_files[0]) as csv_file:
                return _load_csv(csv_file, **kwargs)
    except Exception as e:
        st.error(f"Erro ao processar o arquivo .zip: {e}")
        return None


def load_data(uploaded_file, **kwargs) -> pd.DataFrame | None:
    """Carrega dados de um arquivo CSV ou ZIP e retorna um DataFrame."""
    if uploaded_file.name.endswith(".zip"):
        return _load_from_zip(uploaded_file, **kwargs)
    elif uploaded_file.name.endswith(".csv"):
        return _load_csv(uploaded_file, **kwargs)
    st.error("Formato de arquivo não suportado. Por favor, envie um .csv ou .zip.")
    return None


# Configuração da página
st.set_page_config(
    page_title="Análise Inteligente de Dados",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/seu-usuario/projeto-analise-csv",
        "Report a bug": "mailto:eng.gustavo90lima@gmail.com",
        "About": """
        # Sistema de Análise Inteligente de Dados
        
        Desenvolvido com:
        - 🤖 LangChain para agentes inteligentes
        - 📊 Plotly para visualizações interativas
        - 🗄️ Supabase para persistência de dados
        - 🎨 Streamlit para interface web
        
        **Versão:** 2.0.0
        **Autor:** Gustavo Lima
        """,
    },
)

# --- Constantes e Funções de Modelo ---

AVAILABLE_MODELS = {
    "Google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ],
}


def get_models() -> Dict[str, List[str]]:
    """Retorna a lista de provedores e o dicionário de modelos."""
    return [model for modelList in AVAILABLE_MODELS.values() for model in modelList]


# --- Funções da Interface (UI) ---


def handle_model_change():
    """
    Callback para lidar com a mudança de modelo.
    Limpa o cache do agente para forçar a recriação com as novas configurações.
    """
    st.session_state.selected_provider = (
        "Groq"
        if st.session_state.get("selected_model") in AVAILABLE_MODELS["Groq"]
        else "Google"
    )
    # Limpa o cache para forçar a recriação do agente com o novo modelo
    get_orchestrator.clear()
    st.session_state.agent_initialized = False
    # Re-inicializa se um dataframe já estiver carregado
    initialize_agent()


def initialize_agent() -> None:
    """
    Inicializa ou reinicializa o agente orquestrador com as configurações
    selecionadas no estado da sessão. Exibe erros na interface se a
    inicialização falhar.
    """
    if st.session_state.get("current_dataframe") is None or st.session_state.get(
        "agent_initialized", False
    ):
        return

    try:
        # O agente é inicializado usando a memória do session_state
        st.session_state.orchestrator = get_orchestrator()
        st.session_state.agent_initialized = True
        st.toast(
            f"Agente inicializado com o modelo '{st.session_state.selected_model}'!"
        )
    except Exception as e:
        st.error(f"Falha ao inicializar o agente: {e}")
        st.session_state.agent_initialized = False


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> OrchestratorAgent:
    """
    Cria e armazena em cache o agente orquestrador.
    A decoração @st.cache_resource garante que o agente não seja recriado
    a cada interação do usuário, apenas quando o cache é limpo (ex: troca de modelo).
    A memória é passada explicitamente para garantir que o estado da conversa seja
    consistente com o st.session_state.
    """
    # Passa a memória do session_state para o agente
    return OrchestratorAgent()


def render_sidebar() -> None:
    """Renderiza a barra lateral com as opções de configuração."""
    with st.sidebar:
        st.title("⚙️ Configurações")
        st.markdown("Selecione o provedor e o modelo de linguagem a ser utilizado.")

        # Seleção de Modelo baseada no Provedor
        st.selectbox(
            "Modelo",
            get_models(),
            key="selected_model",
            on_change=handle_model_change,
        )


def render_file_upload() -> None:
    """Renderiza a seção de upload de arquivos."""
    st.header("1. Carregue seu arquivo CSV")
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV ou ZIP para análise", type=["csv", "zip"]
    )

    if uploaded_file is not None:
        # Inicializa o serviço de dados e carrega o arquivo
        dataframe = load_data(uploaded_file)
        if dataframe is not None:
            # Se o dataframe for novo, reseta o estado do agente e do chat
            if not dataframe.equals(st.session_state.get("current_dataframe")):
                st.session_state.current_dataframe = dataframe
                st.session_state.agent_initialized = False
                # Limpa o histórico do chat e a memória do agente para o novo arquivo
                st.session_state.orchestrator_memory.clear()
                st.session_state.chat_history = []
                st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
                st.dataframe(st.session_state.current_dataframe.head())
                initialize_agent()


def render_chat_interface() -> None:
    """Renderiza a interface de chat para interação com o agente."""
    st.header("2. Converse com o Agente")

    # Condição corrigida para evitar o AttributeError
    if st.session_state.get("current_dataframe") is None:
        st.info("Por favor, carregue um arquivo CSV para iniciar a conversa.")
        return

    # Exibe o histórico do chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True)

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre seus dados..."):
        # Adiciona a mensagem do usuário ao histórico e à tela
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa a pergunta com o agente
        with st.chat_message("assistant"):
            with st.spinner("🤖 Pensando..."):
                charts_before = len(st.session_state.charts)
                try:
                    response = st.session_state.orchestrator.process_user_query(prompt)
                    response_content = response.get(
                        "output", "Não consegui processar sua pergunta."
                    )
                    st.markdown(response_content)

                    charts_after = len(st.session_state.charts)

                    # Adiciona a resposta do agente ao histórico
                    new_chart = None
                    if charts_after > charts_before:
                        new_chart = st.session_state.charts[-1]
                        st.plotly_chart(new_chart, use_container_width=True)

                    msg = {"role": "assistant", "content": response_content}
                    if new_chart:
                        msg["chart"] = new_chart
                    st.session_state.chat_history.append(msg)

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_message}
                    )


# --- Função Principal ---


def initialize_session_state() -> None:
    """Inicializa o estado da sessão se não existir."""
    defaults = {
        "selected_model": "gemini-2.5-flash",
        "selected_provider": "Google",
        "chat_history": [],
        "current_dataframe": None,
        "orchestrator": None,
        "charts": [],
        "agent_initialized": False,
        "debug_mode": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # Inicializa a memória do agente separadamente para gerenciamento explícito
    if "orchestrator_memory" not in st.session_state:
        st.session_state.orchestrator_memory = ConversationBufferWindowMemory(
            k=7,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,
        )


def main() -> None:
    """Função principal que renderiza a aplicação Streamlit."""
    st.title("🤖 Análise Inteligente de Dados")

    with st.expander("ℹ️ Como utilizar", expanded=True):
        st.markdown("""
            Bem-vindo ao assistente de análise de dados! Siga os passos abaixo para começar:
            
            1.  **Configure o Agente:** Na barra lateral à esquerda, escolha o provedor (`Groq` ou `Google`) e o modelo de linguagem que deseja usar.
            2.  **Carregue seus Dados:** Utilize a seção de upload para carregar um arquivo no formato `.csv` ou `.zip`.
            3.  **Inicie a Conversa:** Após o carregamento, a janela de chat será habilitada. Faça perguntas em linguagem natural sobre o seu conjunto de dados.
            
            **Exemplos de perguntas:**
            - "Quantas linhas e colunas o dataset possui?"
            - "Mostre um resumo estatístico dos dados."
            - "Crie um gráfico de barras da coluna 'categoria'."
        """)

    initialize_session_state()
    render_sidebar()

    render_file_upload()
    render_chat_interface()


if __name__ == "__main__":
    main()
