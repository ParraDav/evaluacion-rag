from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "../vectorstore"

# ======================================
# EMBEDDINGS
# ======================================
def load_embeddings():
    """Carga el modelo de embeddings de Hugging Face."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ======================================
# VECTOR STORE
# ======================================
def load_vectorstore():
    """Carga la base de datos Chroma persistida."""
    embeddings = load_embeddings()
    
    # Se usa la nueva clase de langchain_chroma
    vectordb = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    return vectordb

# ======================================
# RETRIEVER
# ======================================
def load_retriever(k=3):
    """Configura el recuperador de documentos."""
    vectordb = load_vectorstore()
    return vectordb.as_retriever(
        search_kwargs={"k": k}
    )

# ======================================
# PROMPT
# ======================================
def load_prompt():
    """Define la estructura del prompt para el modelo."""
    template = """
    Responde únicamente usando el contexto proporcionado.
    Si la respuesta no está en el contexto, responde:
    "No encontré información suficiente en los documentos."

    Contexto:
    {context}

    Pregunta:
    {question}
    """
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# ======================================
# MOSTRAR CHUNKS
# ======================================
def print_chunks(source_documents):
    """Imprime los fragmentos recuperados para depuración."""
    print("\n--- Chunks recuperados ---")
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get("source", "Desconocido")
        page = doc.metadata.get("page", "?")
        print(f"Chunk {i+1} [Origen: {source} - Pág: {page}]:")
        print(f"{doc.page_content[:300]}...")
        print("-" * 30)