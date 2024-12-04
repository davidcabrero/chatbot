import os
import ollama
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import fitz  # PyMuPDF
from PIL import Image
import tempfile

# Configuración de los modelos de lenguaje
llm_text = OllamaLLM(model="llama3.2:1b", temperature=0.2)

# Función para pasar consulta con imagen
def consultaImagen(image_path, user_input):
    response = ollama.chat(
        model='llava',
        messages=[{
            'role': 'user',
            'content': user_input,
            'images': [image_path]
        }]
    )
    return response

# Función para extraer texto de un PDF usando PyMuPDF
def extract_pdf_text(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Resumir el historial de chat para reducir datos enviados al modelo
def summarize_chat_history(chat_history, max_length=10):
    """Resumen del historial de chat para reducir el tamaño de los datos enviados al modelo."""
    if len(chat_history) > max_length:  # Limitar el historial a las últimas interacciones
        summarized = [msg for msg in chat_history[-max_length:]]
        return summarized
    return chat_history

def main():
    # Configurar la aplicación de Streamlit
    st.set_page_config(page_title="ChatBot", layout="wide")
    st.image("logo.png", width=150)

    # Configuración del chatbot
    bot_name = "ChatBot"
    bot_prompt = f"Eres una IA desarrollada por David Cabrero y te llamas {bot_name}. Respondes y haces lo que te pida el usuario."

    # Historial de chat
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Variables para almacenar contenido
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""

    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([ 
        ("system", bot_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])

    cadena = prompt_template | llm_text

    # Sugerencias de preguntas
    st.markdown("### Preguntas Sugeridas")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("¿Cuál es la capital de Francia?"):
            st.session_state["user_input"] = "¿Cuál es la capital de Francia?"
    with col2:
        if st.button("Muestra un gráfico de ejemplo"):
            st.session_state["user_input"] = "Muestra un gráfico con datos [A, B, C] [10, 20, 15]"
    with col3:
        if st.button("¿Quién te creó?"):
            st.session_state["user_input"] = "¿Quién te creó?"
    with col4:
        if st.button("Programa Python"):
            st.session_state["user_input"] = "Programa en python la suma de 2 números"

    # Subir PDF o imagen
    st.markdown("### Subir Archivos")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_pdf = st.file_uploader("Sube un archivo PDF", type="pdf")
    with col2:
        uploaded_image = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_pdf is not None:
        # Guardar el archivo PDF temporalmente en el disco
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            pdf_file_path = tmp_file.name  # Ruta completa del archivo temporal

        # Extraer el texto del PDF
        st.session_state["pdf_text"] = extract_pdf_text(pdf_file_path)
        st.write("PDF cargado con éxito. Puedes hacer preguntas sobre su contenido.")

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())
            image_file_path = tmp_file.name  # Ruta completa del archivo temporal
        # Guardar la imagen subida
        st.session_state["uploaded_image"] = Image.open(image_file_path)
        st.write("Imagen cargada con éxito. Puedes hacer preguntas sobre ella.")

    # Entrada de usuario
    user_input = st.text_input("Escribe tu pregunta", key="user_input")

    # Sección de Contacto o Ayuda
    st.sidebar.title("ChatBot - Menú")
    st.sidebar.write("Explora las funcionalidades de ChatBot o contacta con soporte.")
    st.sidebar.header("Acerca de ChatBot")
    st.sidebar.write("ChatBot es una IA desarrollada por David Cabrero.")
    st.sidebar.header("Contacto")
    st.sidebar.write("Para más información, contáctanos en davidcabrerojimenez@gmail.com")

    if st.button("Preguntar"):
        if user_input.lower() == "adios":
            st.stop()
        else:
            respuesta = ""
            # Resumir historial de chat antes de la consulta
            current_history = summarize_chat_history(st.session_state["chat_history"])

            if "uploaded_image" in st.session_state and 'image_file_path' in locals():
                # Procesar pregunta relacionada con la imagen usando LLaVA
                respuesta_imagen = consultaImagen(image_file_path, user_input)
                if "message" in respuesta_imagen and "content" in respuesta_imagen["message"]:
                    respuesta = respuesta_imagen["message"]["content"]
                else:
                    respuesta = "No se pudo obtener una respuesta válida para la imagen."
            elif "pdf_text" in st.session_state and st.session_state["pdf_text"]:
                # Preguntas relacionadas con el PDF
                user_input_pdf = f"Texto: {st.session_state['pdf_text']}\nPregunta: {user_input}"
                respuesta_texto = cadena.invoke({"user_input": user_input_pdf, "chat_history": current_history})
                respuesta = respuesta_texto
            else:
                # Pregunta estándar
                respuesta_texto = cadena.invoke({"user_input": user_input, "chat_history": current_history})
                respuesta = respuesta_texto

            # Agregar al historial
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=respuesta))

            # Detectar si el usuario pidió un gráfico y especificó datos
            if "gráfico" in user_input.lower():
                datos = re.findall(r"\[(.*?)\]", user_input)
                if datos:
                    categorias = datos[0].split(',')
                    valores = list(map(float, datos[1].split(','))) if len(datos) > 1 else []

                    if len(categorias) == len(valores) and len(categorias) > 0:
                        # Crear un gráfico de barras con los datos proporcionados
                        st.write("Generando gráfico solicitado...")

                        fig, ax = plt.subplots()
                        ax.bar(categorias, valores)
                        ax.set_title('Gráfico Personalizado')
                        ax.set_xlabel('Categoría')
                        ax.set_ylabel('Valores')

                        # Mostrar el gráfico en la aplicación
                        st.pyplot(fig)
                    else:
                        st.write("Error: Asegúrate de que las categorías y valores coincidan en número y formato.")
                else:
                    st.write("Error: Por favor, ingresa los datos en el formato [categoría1, categoría2, ...] [valor1, valor2, ...].")

    # Mostrar historial de chat en burbujas de conversación diferenciando Usuario y ChatBot
    st.markdown("### Chat")
    for mensaje in st.session_state["chat_history"]:
        if isinstance(mensaje, HumanMessage):
            with st.chat_message("Usuario"):
                if "Texto:" in mensaje.content:
                    st.write(mensaje.content.split("\nPregunta:")[1].strip())  # Mostrar solo la pregunta
                else:
                    st.write(mensaje.content)
        elif isinstance(mensaje, AIMessage):
            with st.chat_message(bot_name):
                st.write(mensaje.content)

if __name__ == "__main__":
    main()