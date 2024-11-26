import ollama
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import fitz  # PyMuPDF

# Configuración del modelo de lenguaje
llm = OllamaLLM(model="llama3.2:1b", temperature=0.2)

# Función para extraer texto de un PDF usando PyMuPDF
def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

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
    
    # Variable para almacenar el contenido del PDF
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""

    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", bot_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])

    cadena = prompt_template | llm

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

    # Subir PDF
    uploaded_pdf = st.file_uploader("Sube un archivo PDF", type="pdf")

    if uploaded_pdf is not None:
        # Extraer el texto del PDF
        st.session_state["pdf_text"] = extract_pdf_text(uploaded_pdf)
        st.write("PDF cargado con éxito. Puedes hacer preguntas sobre su contenido.")

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
            # Si se ha subido un PDF, incluir el contenido del PDF en el prompt
            if st.session_state["pdf_text"]:
                user_input = f"Texto del PDF: {st.session_state['pdf_text']}\nPregunta: {user_input}"

            # Procesar la solicitud del usuario
            respuesta = cadena.invoke({"user_input": user_input, "chat_history": st.session_state["chat_history"]})
            
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
                st.write(mensaje.content)
        elif isinstance(mensaje, AIMessage):
            with st.chat_message(bot_name):
                st.write(mensaje.content)

if __name__ == "__main__":
    main()