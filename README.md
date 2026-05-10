# Evaluación de un Sistema RAG con LangChain y Ollama

## Descripción

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) utilizando:

- LangChain
- ChromaDB
- HuggingFace Embeddings
- Ollama
- RAGAS

El sistema carga documentos PDF, los divide en fragmentos (chunks), genera embeddings y almacena la información en una base vectorial para posteriormente responder preguntas sobre el contenido utilizando un modelo local ejecutado con Ollama.

También se realiza una evaluación automática del sistema utilizando métricas de RAGAS.

---

# Instalación

## 1. Crear entorno virtual

```bash
python -m venv venv
2. Activar entorno virtual
Windows PowerShell
venv\Scripts\activate
Instalar dependencias
pip install -r requirements.txt
Instalar Ollama

Descargar e instalar:

https://ollama.com

Verificar instalación:

ollama --version
Descargar modelo Llama3
ollama pull llama3

Verificar modelos instalados:

ollama list
Agregar documentos

Colocar los archivos PDF dentro de:

data/

Ejemplo:

data/reglamento.pdf
Generar embeddings y base vectorial
python src/ingest.py

Este proceso:

carga los PDFs,
divide el texto en chunks,
genera embeddings,
crea la base vectorial ChromaDB.
Ejecutar el pipeline RAG
python src/rag_pipeline.py

El sistema responderá preguntas utilizando el contenido de los documentos cargados.

Ejecutar evaluación con RAGAS
python src/evaluate_ragas.py

La evaluación genera métricas como:

Faithfulness
Answer Relevancy
Context Precision

Los resultados se guardan en:

results/ragas_results.csv
Resultado esperado

Ejemplo:

Pregunta: ¿Qué ocurre si un estudiante supera el 20% de inasistencia?

Respuesta:
La asignatura se califica con cero (0) y se registra como reprobada por inasistencia.