import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Cambio: Ahora se usa langchain_chroma
from langchain_chroma import Chroma

DATA_PATH = "../data"
VECTORSTORE_PATH = "../vectorstore"

# =========================
# CARGAR DOCUMENTOS
# =========================
documents = []
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Documentos cargados: {len(documents)}")

# =========================
# CHUNKING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Chunks generados: {len(chunks)}")

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR STORE
# =========================
# Nota: En langchain_chroma ya no es necesario llamar a .persist(), 
# se guarda automáticamente al inicializar.
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTORSTORE_PATH
)

print("Base vectorial creada y guardada correctamente.")