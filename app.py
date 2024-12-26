__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configurar a página
st.set_page_config(
    page_title="Chatbot Chácara-MG",
    page_icon="🏘️",
    layout="centered"
)

# Classe de embedding (mesma que você já tem)
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=embedding_task,
            request_options=retry_policy,
        )
        return response["embedding"]

# Função para carregar dados
@st.cache_resource
def load_knowledge_base():
    data = pd.read_csv("data/chácara_knowledge.csv", 
                      encoding='latin1',
                      sep=',',
                      quotechar='"',
                      escapechar='\\',
                      on_bad_lines='warn')
    return data

# Função para inicializar ChromaDB
@st.cache_resource
def init_chromadb():
    embed_fn = GeminiEmbeddingFunction()
    
    # Usar o novo cliente persistente
    #chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Usar o cliente em memória
    chroma_client = chromadb.Client()
    
    db = chroma_client.get_or_create_collection(
        name="characadb",
        embedding_function=embed_fn
    )
    
    # Carregar documentos se o DB estiver vazio
    if db.count() == 0:
        data = load_knowledge_base()
        documents = data['ConteÃºdo'].tolist()
        db.add(
            documents=documents,
            ids=[str(i) for i in range(len(documents))]
        )
    
    return db, embed_fn

# Inicializar o modelo Gemini
@st.cache_resource
def init_model():
    SYSTEM_MESSAGE = '''Você é um agente especializado em dar informações sobre a cidade Chácara em Minas Gerais.
    Você é um agente que responde sobre questões históricas, culturais e geografia. 
    Você não responderá questões que não sejam sobre chácara e caso alguém pergunte, 
    você responderá educadamente que não pode dar estas informações.
    '''
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        system_instruction=SYSTEM_MESSAGE
    )
    
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )
    
    return chat


# Interface principal
def main():
    
    st.title("🏘️ Chatbot Chácara-MG")
    
    # Adicionar exemplos de perguntas antes do chat input
    st.markdown("""
    ### Exemplos de perguntas que você pode fazer:
    - 🏛️ "Qual é a história de Chácara?"
    - 🚌 "Quais os horários de ônibus de CH vs. JF durante a semana de manhã?"
    - 🌍 "Me fale sobre a geografia da cidade.."
    - 🎭 "Quais são as principais festas da cidade?"
    - 👥 "Quantos habitantes tem Chácara?"
    - 🍽️ "Quais os restaurantes da cidade?"
    """)
    
    # Seção expansível com exemplos
    with st.expander("📝 Clique aqui para ver exemplos de perguntas"):
        st.markdown("""
        ### Temas que você pode explorar:
        
        **História e Cultura:**
        - História da fundação da cidade
        - Origem do nome
        - Tradições locais
        
        **Geografia e Demografia:**
        - Localização
        - População
        - Clima
        
        **Turismo:**
        - Pontos turísticos
        - Festas tradicionais
        - Gastronomia local
        
        **Economia:**
        - Principais atividades econômicas
        - Produtos locais
        - Comércio
        """)
    
    # Inicializar componentes
    db, embed_fn = init_chromadb()
    chat = init_model()
    
    # Inicializar histórico de chat no session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça sua pergunta sobre Chácara-MG"):
        # Adicionar pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                # Buscar contexto relevante
                embed_fn.document_mode = False
                result = db.query(
                    query_texts=[prompt],
                    n_results=2
                )
                context = "\n".join(result["documents"][0])
                
                # Criar query aumentada
                augmented_query = f"""
                Contexto: {context}
                
                Pergunta do usuário: {prompt}
                
                Por favor, responda à pergunta usando as informações do contexto acima.
                """
                
                # Gerar resposta
                response = chat.send_message(augmented_query)
                st.markdown(response.text)
                
                # Adicionar resposta ao histórico
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text}
                )

    # Sidebar com informações
    with st.sidebar:
        st.markdown("### Sobre")
        st.write("Este chatbot utiliza RAG (Retrieval-Augmented Generation) para fornecer informações precisas sobre a cidade de Chácara-MG.")
        
        if st.button("Limpar Conversa"):
            st.session_state.messages = []
            st.rerun()
            
        

if __name__ == "__main__":
    main()