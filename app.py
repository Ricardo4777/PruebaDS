# Importa las bibliotecas necesarias
from dotenv import load_dotenv
import os
import streamlit as st
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain
import time
import feedparser  # Nueva importación para procesar archivos RSS

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un documento HTML o RSS
def process_content(content, content_type):
    if content_type == "html":
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
    elif content_type == "rss":
        text = " ".join(entry.title + " " + entry.description for entry in feedparser.parse(content).entries)
    else:
        raise ValueError("Tipo de contenido no admitido")

    # Divide el texto en trozos usando langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# Función principal de la aplicación
def main():
    st.title("Preguntas a Documentos HTML y RSS")

    html_file = st.file_uploader("Sube tu documento HTML", type="html")
    rss_file = st.file_uploader("Sube tu archivo RSS", type="rss")

    if html_file is not None and rss_file is not None:
        html_content = html_file.read()
        rss_content = rss_file.read()

        # Procesa el contenido HTML y RSS para extraer texto y generar bases de conocimientos
        knowledge_base_html = process_content(html_content, "html")
        knowledge_base_rss = process_content(rss_content, "rss")

        # Combina las listas de documentos de HTML y RSS


        query = st.text_input('Escribe tu pregunta para los documentos HTML y RSS...')


        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()

        if query:
            # Realiza una búsqueda de similitud en la base de conocimientos combinada
            result_docs = knowledge_base_html.similarity_search(query)
            result_docs = knowledge_base_rss.similarity_search(query)

            # Inicializa un modelo de lenguaje de OpenAI y ajusta sus parámetros
            model = "gpt-3.5-turbo-instruct"  # Acepta 4096 tokens
            temperature = 0  # Valores entre 0 - 1
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            # Carga la cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="map_reduce")
            # Inicia el temporizador justo antes de realizar la llamada a la API de OpenAI
            start_time = time.time()

            # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": result_docs})

            # Calcula el tiempo de transacción en segundos después de recibir la respuesta
            end_time = time.time()
            transaction_time = end_time - start_time

            # Imprime el costo de la operación y el tiempo de transacción
            print(cost)
            print("Tiempo de transacción:", transaction_time, "segundos")

            # Muestra los resultados en la interfaz de usuario
            st.write(f"Tiempo de transacción: {transaction_time} segundos")
            st.write(f"Costo de tokens: {cost}")

            st.write(response["output_text"])

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()
