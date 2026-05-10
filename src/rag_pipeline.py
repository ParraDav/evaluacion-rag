from datasets import Dataset

from langchain_ollama import ChatOllama
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

llm = ChatOllama(
    model="llama3",
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
# PREGUNTAS
# ======================================

questions = [
    "¿Cuál es la nota mínima aprobatoria de una asignatura?", 
"¿Cuántas inasistencias puede tener un estudiante antes de perder una materia?", 
"¿Qué ocurre si un estudiante reprueba una materia?", 
"¿Cómo se obtiene el promedio general del semestre?", 
"¿Qué consecuencias académicas y administrativas tiene cancelar una asignatura?", 
"¿Qué requisitos debe cumplir un estudiante para homologar materias y cómo afecta eso su historial académico?", 
"¿Cuál es el color oficial del uniforme institucional?", 
"¿Qué marca de computadores recomienda la universidad para los estudiantes?"

]

# ======================================
# DATASET PARA RAGAS
# ======================================

data = {
    "question": [],
    "answer": [],
    "contexts": []
}

for question in questions:

    print(f"\nPregunta: {question}")

    result = qa_chain.invoke({"query": question})

    answer = result["result"]

    print(f"Respuesta: {answer}")

    contexts = [
        doc.page_content
        for doc in result["source_documents"]
    ]

    data["question"].append(question)
    data["answer"].append(answer)
    data["contexts"].append(contexts)

dataset = Dataset.from_dict(data)

# ======================================
# EVALUACIÓN RAGAS
# ======================================

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=llm
)

df = result.to_pandas()

print("\nResultados:")
print(df)

df.to_csv("../results/ragas_results.csv", index=False)

print("\nResultados guardados en results/ragas_results.csv")