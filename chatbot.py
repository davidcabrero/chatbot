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
from gtts import gTTS  # Para la generación de voz
from diffusers import StableDiffusionPipeline

# Configuración del modelo de lenguaje
llm_text = OllamaLLM(model="llama3.2:1b", temperature=0.2)

# Función para consultar con imagen (usando LLaVA)
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

# Función para resumir el historial de chat
def summarize_chat_history(chat_history, max_length=10):
    if len(chat_history) > max_length:
        return chat_history[-max_length:]
    return chat_history

# Cargar el modelo de Stable Diffusion solo si se requiere
def load_image_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    return model.to("cpu")  # Cambia a 'cuda' si tienes una GPU compatible

# Función para generar una imagen
def generate_image(prompt, sd_model):
    with st.spinner("Generando imagen..."):
        image = sd_model(prompt).images[0]
        return image

def main():
    # Configuración inicial de Streamlit
    st.set_page_config(page_title="ChatBot", layout="wide")
    st.image("logo.png", width=150)  # Reemplaza con tu logo

    # Variables de estado
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""
    if "image_model" not in st.session_state:
        st.session_state["image_model"] = None

    # Configuración del chatbot
    bot_name = "ChatBot"
    bot_prompt = f"Eres una IA desarrollada por David Cabrero y te llamas {bot_name}. Respondes y haces lo que te pida el usuario."
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", bot_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])
    cadena = prompt_template | llm_text

    # Preguntas sugeridas
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

    # Subir archivos (PDF o imagen)
    st.markdown("### Subir Archivos")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_pdf = st.file_uploader("Sube un archivo PDF", type="pdf")
    with col2:
        uploaded_image = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            pdf_file_path = tmp_file.name
        st.session_state["pdf_text"] = extract_pdf_text(pdf_file_path)
        st.write("PDF cargado con éxito. Puedes hacer preguntas sobre su contenido.")

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())
            image_file_path = tmp_file.name
        st.session_state["uploaded_image"] = Image.open(image_file_path)
        st.write("Imagen cargada con éxito. Puedes hacer preguntas sobre ella.")

    # Entrada del usuario
    user_input = st.text_input("Escribe tu pregunta", key="user_input")

    # Barra lateral
    st.sidebar.title("ChatBot - Menú")
    st.sidebar.write("Explora las funcionalidades de ChatBot o contacta con soporte.")
    st.sidebar.header("Acerca de ChatBot")
    st.sidebar.write("ChatBot es una IA desarrollada por David Cabrero.")
    st.sidebar.header("Contacto")
    st.sidebar.write("Para más información, contáctanos en davidcabrerojimenez@gmail.com")

    if st.button("Preguntar"):
        respuesta = ""
        current_history = summarize_chat_history(st.session_state["chat_history"])

        if "uploaded_image" in st.session_state and 'image_file_path' in locals():
            respuesta_imagen = consultaImagen(image_file_path, user_input)
            respuesta = respuesta_imagen.get("message", {}).get("content", "No se pudo procesar la imagen.")
        elif "pdf_text" in st.session_state and st.session_state["pdf_text"]:
            user_input_pdf = f"Texto: {st.session_state['pdf_text']}\nPregunta: {user_input}"
            respuesta = cadena.invoke({"user_input": user_input_pdf, "chat_history": current_history})
        elif "genera una imagen de" in user_input.lower():
            if not st.session_state["image_model"]:
                st.session_state["image_model"] = load_image_model()
                # Extraer el prompt de la imagen
                prompt_image = user_input.lower().replace("genera una imagen de", "").strip()
                # Generar imagen
                image = generate_image(prompt_image, st.session_state["image_model"])
                # Guardar la imagen temporalmente para mostrarla
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    image_path = tmp_file.name
                # Mostrar la imagen
                st.image(image, caption=f"Imagen generada: {prompt_image}", use_container_width=True)
                respuesta = f"He generado una imagen para: {prompt_image}"    
        else:
            respuesta = cadena.invoke({"user_input": user_input, "chat_history": current_history})

        st.session_state["chat_history"].append(HumanMessage(content=user_input))
        st.session_state["chat_history"].append(AIMessage(content=respuesta))

        # Detectar si el usuario pidió un gráfico
        if "gráfico" in user_input.lower():
            datos = re.findall(r"\[(.*?)\]", user_input)
            if datos:
                categorias = datos[0].split(',')
                valores = list(map(float, datos[1].split(','))) if len(datos) > 1 else []
                if len(categorias) == len(valores) and categorias:
                    fig, ax = plt.subplots()
                    ax.bar(categorias, valores)
                    ax.set_title('Gráfico Personalizado')
                    ax.set_xlabel('Categorías')
                    ax.set_ylabel('Valores')
                    st.pyplot(fig)
                else:
                    st.write("Error: Categorías y valores deben coincidir.")
            else:
                st.write("Error: Usa el formato [categoría1, categoría2] [valor1, valor2].")

    # Mostrar historial de chat
    st.markdown("### Chat")
    for mensaje in st.session_state["chat_history"]:
        role = "Usuario" if isinstance(mensaje, HumanMessage) else bot_name
        with st.chat_message(role):
            st.write(mensaje.content)

    # Generar audio para la última respuesta
    if st.session_state["chat_history"] and isinstance(st.session_state["chat_history"][-1], AIMessage):
        audio_file = "respuesta.mp3"
        if os.path.exists(audio_file):
            os.remove(audio_file)
        tts = gTTS(text=st.session_state["chat_history"][-1].content, lang='es')
        tts.save(audio_file)
        with open(audio_file, "rb") as f:
            st.audio(f.read(), format="audio/mp3")

if __name__ == "__main__":
    main()
