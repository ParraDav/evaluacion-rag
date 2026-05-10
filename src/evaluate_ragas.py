from dotenv import load_dotenv
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

from utils import (
    load_retriever,
    load_prompt
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

questions = [
    "¿Cuál es la nota mínima aprobatoria?",
    "¿Qué pasa si un estudiante falta demasiado?",
    "¿Cómo puedo recuperar una materia perdida?",
    "¿Qué requisitos existen para homologación?",
    "¿Cómo afecta cancelar una asignatura al promedio?",
    "¿Cuál es el color oficial del uniforme?"
]

data = {
    "question": [],
    "answer": [],
    "contexts": []
}

for question in questions:

    result = qa_chain.invoke({"query": question})

    data["question"].append(question)

    data["answer"].append(result["result"])

    data["contexts"].append([
        doc.page_content
        for doc in result["source_documents"]
    ])

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)

df = result.to_pandas()

print(df)

df.to_csv("../results/ragas_results.csv", index=False)