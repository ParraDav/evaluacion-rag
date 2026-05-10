from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from utils import (
    load_retriever,
    load_prompt,
    print_chunks
)

load_dotenv()

# ======================================
# RETRIEVER
# ======================================

retriever = load_retriever(k=3)

# ======================================
# PROMPT
# ======================================

prompt = load_prompt()

# ======================================
# LLM
# ======================================

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0
)

# ======================================
# QA CHAIN
# ======================================

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt
    }
)

# ======================================
# CONSULTAS
# ======================================

while True:

    query = input("\nPregunta: ")

    if query.lower() == "salir":
        break

    result = qa_chain.invoke({"query": query})

    print("\nRespuesta:\n")
    print(result["result"])

    print_chunks(result["source_documents"])