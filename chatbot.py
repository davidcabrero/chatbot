# David Cabrero Jiménez
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
import torch
import plotly.express as px
from fpdf import FPDF
from spacy import displacy
import spacy
from textblob import TextBlob
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import agentes
import os

# Descargar datos de nltk si no están ya descargados
nltk.download('punkt')

spacy_model = spacy.load("en_core_web_sm")

# Configuración del modelo de lenguaje
llm_text = OllamaLLM(model="llama3.2:1b", temperature=0.2)

# Función para gestionar tareas
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []

def add_task(task_name, due_date, assigned_to):
    st.session_state["tasks"].append({"task": task_name, "due_date": due_date, "assigned_to": assigned_to, "completed": False})

def mark_task_completed(index):
    st.session_state["tasks"][index]["completed"] = True

def delete_task(index):
    st.session_state["tasks"].pop(index)

# Función para crear dashboards interactivos
def create_dashboard(data):
    st.markdown("## Análisis de Datos")
    chart_type = st.selectbox("Selecciona el tipo de gráfico", ["Barras", "Dispersión", "Líneas"])
    x_axis = st.selectbox("Selecciona el eje X", data.columns)
    y_axis = st.selectbox("Selecciona el eje Y", data.columns)

    if chart_type == "Barras":
        fig = px.bar(data, x=x_axis, y=y_axis, title="Gráfico de Barras")
    elif chart_type == "Dispersión":
        fig = px.scatter(data, x=x_axis, y=y_axis, title="Gráfico de Dispersión")
    elif chart_type == "Líneas":
        fig = px.line(data, x=x_axis, y=y_axis, title="Gráfico de Líneas")

    st.plotly_chart(fig)
    return chart_type, x_axis, y_axis

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

# Función para cargar el modelo de Stable Diffusion
def load_image_model():
    if "image_model" not in st.session_state or st.session_state["image_model"] is None:
        st.session_state["image_model"] = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        st.session_state["image_model"] = st.session_state["image_model"].to("cuda" if torch.cuda.is_available() else "cpu")
    return st.session_state["image_model"]

# Función para generar una imagen
def generate_image(prompt):
    sd_model = load_image_model()
    with torch.no_grad():  # Desactivar gradientes para inferencia (aumenta velocidad)
        with torch.cuda.amp.autocast():  # Precisión mixta
            with st.spinner("Generando imagen..."):
                image = sd_model(prompt).images[0]
                return image

# Función para formatear los datos del CSV en la estructura especificada
def format_data(data):
    columns = data.columns.tolist()
    rows = data.values.tolist()
    formatted_data = f"Nombres Columnas: {columns}\nFilas:\n"
    for row in rows:
        formatted_data += f"[{', '.join(map(str, row))}]\n"
    return formatted_data

# Función para generar un PDF con la última respuesta del chatbot
def generate_pdf(text, filename="respuesta.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)

# Funciones adicionales usando Spacy, TextBlob, Gensim, NLTK, y CrewAI
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment

def tokenize_text(text):
    return word_tokenize(text)

def named_entity_recognition(text):
    doc = spacy_model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def main():
    # Configuración inicial de Streamlit
    st.set_page_config(page_title="DataBot", layout="wide")
    st.image("logo.png", width=150)

    # Variables de estado
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "pdf_text" not in st.session_state:
        st.session_state["pdf_text"] = ""
    if "uploaded_image_path" not in st.session_state:
        st.session_state["uploaded_image_path"] = None
    if "csv_data" not in st.session_state:
        st.session_state["csv_data"] = None
    if "chart_info" not in st.session_state:
        st.session_state["chart_info"] = None
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None    

    # Configuración del chatbot
    bot_name = "DataBot"
    bot_prompt = f"Eres una IA desarrollada por David Cabrero, enfocada a los analistas de datos y te llamas {bot_name}. Respondes y haces lo que te pida el usuario."
    prompt_template = ChatPromptTemplate.from_messages([ 
        ("system", bot_prompt), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])
    cadena = prompt_template | llm_text

    # Barra lateral
    st.sidebar.title("DataBot - Menú")
    menu = st.sidebar.selectbox("Selecciona una funcionalidad", ["Chat", "Gestión de Tareas", "Análisis de Datos", "Tools", "Web Scraping"])

    # Botón para borrar el historial de chat y dejar vacio la sección de los mensajes
    if st.sidebar.button("Borrar historial"):
        st.session_state["chat_history"] = []
        st.session_state["pdf_text"] = ""
        st.session_state["uploaded_image_path"] = None
        st.session_state["csv_data"] = None
        st.session_state["chart_info"] = None
        st.session_state["uploaded_file"] = None

    # Botón para descargar el PDF con la última respuesta del chatbot
    if st.sidebar.button("Generar PDF de respuesta"):
        if st.session_state["chat_history"] and isinstance(st.session_state["chat_history"][-1], AIMessage):
            last_response = st.session_state["chat_history"][-1].content
            generate_pdf(last_response)
            with open("respuesta.pdf", "rb") as f:
                st.sidebar.download_button("Descargar PDF", f, file_name="respuesta.pdf")    

    # Funcionalidades
    if menu == "Gestión de Tareas":
        st.markdown("## Gestión de Tareas")

        col1, col2, col3 = st.columns(3)
        with col1:
            task_name = st.text_input("Nombre de la tarea")
        with col2:
            due_date = st.date_input("Fecha de vencimiento")
        with col3:
            assigned_to = st.text_input("Asignar a")

        if st.button("Agregar Tarea"):
            if task_name and assigned_to:
                add_task(task_name, due_date, assigned_to)
                st.success("Tarea agregada correctamente.")

        st.markdown("### Lista de Tareas")
        for i, task in enumerate(st.session_state["tasks"]):
            task_status = "\u2705" if task["completed"] else "\u274C"
            col1, col2, col3, col4 = st.columns([5, 2, 2, 1])
            col1.write(f"{task_status} {task['task']} (Vence: {task['due_date']}) - Asignada a: {task['assigned_to']}")
            if not task["completed"]:
                col2.button("Completar", key=f"complete_{i}", on_click=mark_task_completed, args=(i,))
            col3.button("Eliminar", key=f"delete_{i}", on_click=delete_task, args=(i,))

    elif menu == "Análisis de Datos":
        st.markdown("## Análisis de Datos")
        st.markdown("Sube un archivo CSV para crear un dashboard interactivo.")
        uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.session_state["csv_data"] = data
            st.write("Vista previa de los datos:")
            st.dataframe(data.head())
            chart_info = create_dashboard(data) # Crear las gráficas
            st.session_state["chart_info"] = chart_info

        if st.session_state["csv_data"] is not None:
            st.markdown("### Preguntas sobre el CSV")
            user_input_csv = st.text_input("Escribe tu pregunta sobre el CSV", key="user_input_csv")

            if st.button("Preguntar sobre CSV"):
                respuesta = ""
                current_history = summarize_chat_history(st.session_state["chat_history"])

                csv_table = format_data(st.session_state["csv_data"])
                user_input_csv_prompt = f"Con estos datos:\n{csv_table}\nresponde a esta pregunta: {user_input_csv}"
                respuesta = cadena.invoke({"user_input": user_input_csv_prompt, "chat_history": current_history})

                st.session_state["chat_history"].append(HumanMessage(content=user_input_csv))
                st.session_state["chat_history"].append(AIMessage(content=respuesta))

            # Mostrar historial de chat para preguntas sobre el CSV
            st.markdown("### Chat sobre CSV")
            for mensaje in st.session_state["chat_history"]:
                role = "Usuario" if isinstance(mensaje, HumanMessage) else bot_name
                with st.chat_message(role):
                    st.write(mensaje.content)              

    elif menu == "Tools":
        st.markdown("## Procesamiento de Lenguaje Natural")
        user_input_nlp = st.text_area("Escribe tu texto aquí")

        if st.button("Análisis de Sentimiento"):
            sentiment = sentiment_analysis(user_input_nlp)
            st.write(f"Análisis de Sentimiento: {sentiment}")

        if st.button("Tokenizar Texto"):
            tokens = tokenize_text(user_input_nlp)
            st.write(f"Tokens: {tokens}")

        if st.button("Reconocimiento de Entidades Nombradas"):
            entities = named_entity_recognition(user_input_nlp)
            st.write(f"Entidades Nombradas: {entities}")

    elif menu == "Web Scraping":
        st.markdown("## Web Scraping")
        
        pregunta = st.text_input("Introduce tu pregunta:")
        url = st.text_input("Introduce la URL de la página a analizar:")
        
        if st.button("Ejecutar Web Scraping"):
            if pregunta and url:
                resultados = agentes.web_scraping_tool(pregunta, url)
                st.write("### Resultados del scraping:")
                st.text(resultados)
            else:
                st.warning("Por favor, introduce una pregunta y una URL válida.")

    elif menu == "DataAgents":
        st.markdown("## DataAgents")

    elif menu == "Chat":

        st.markdown("### Preguntas Sugeridas")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("¿Cuál es la capital de Francia?"):
                st.session_state["user_input"] = "¿Cuál es la capital de Francia?"
        with col2:
            if st.button("Gráfico de ejemplo"):
                st.session_state["user_input"] = "Muestra un gráfico con datos [A, B, C] [10, 20, 15]"
        with col3:
            if st.button("¿Quién te creó?"):
                st.session_state["user_input"] = "¿Quién te creó?"
        with col4:
            if st.button("Programa Python"):
                st.session_state["user_input"] = "Programa en python la suma de 2 números"
        with col5:
            if st.button("Genera imagen"):
                st.session_state["user_input"] = "Genera una imagen de un gato" 
        with col6:
            if st.button("Tareas activas"):
                st.session_state["user_input"] = "Tareas activas"                        

        st.markdown("### Subir Archivos")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_pdf = st.file_uploader("Sube un archivo PDF", type="pdf", key="uploaded_pdf")
        with col2:
            uploaded_image = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"], key="uploaded_image")
        if st.session_state["uploaded_file"] and not uploaded_file:
           st.write("Has eliminado el archivo subido.")

        use_agent = st.checkbox("Usar Agente")
        if use_agent:
            agente = st.selectbox("Selecciona un agente", ["CodeAgent", "WriteAgent", "TranslateAgent", "DataGenAgent"])

        # Procesar archivos subidos   
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                pdf_file_path = tmp_file.name
            st.session_state["pdf_text"] = extract_pdf_text(pdf_file_path)
            st.write("PDF cargado con éxito. Puedes hacer preguntas sobre su contenido.")

        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_image.read())
                st.session_state["uploaded_image_path"] = tmp_file.name
            st.write("Imagen cargada con éxito. Puedes hacer preguntas sobre ella.")

        user_input = st.text_input("Escribe tu pregunta", key="user_input")

        use_internet = st.checkbox("Usar búsqueda en internet para esta pregunta")

        if st.button("Preguntar"):

            respuesta = ""

            # Preguntas con búsqueda en internet
            if use_internet:
                st.write("Realizando búsqueda en internet...")
                search_results = agentes.internet_search_tool(user_input)
                
                # Formatear los resultados de la búsqueda para mostrarlos de manera más legible
                results_str = ""
                for i, item in enumerate(search_results, start=1):
                    title = item.get("title", "Sin título")
                    link = item.get("url", "Sin enlace")
                    snippet = item.get("snippet", "Sin descripción")
                    results_str += f"{i}. **{title}**\n   {link}\n   _{snippet}_\n\n"
                
                respuesta = f"Resultados de la búsqueda:\n{results_str}"  
        
            # Preguntas con archivos
            elif "uploaded_image_path" in st.session_state and st.session_state["uploaded_image_path"]:
                respuesta_imagen = consultaImagen(st.session_state["uploaded_image_path"], user_input)
                respuesta = respuesta_imagen.get("message", {}).get("content", "No se pudo procesar la imagen.")
            elif "pdf_text" in st.session_state and st.session_state["pdf_text"]:
                user_input = f"Texto: {st.session_state['pdf_text']}\nPregunta: {user_input}"
                if not use_agent:
                    respuesta = cadena.invoke({"user_input": user_input, "chat_history": st.session_state["chat_history"]})
            elif "genera una imagen de" in user_input.lower():
                prompt_image = user_input.lower().replace("genera una imagen de", "").strip()
                image = generate_image(prompt_image)
                st.image(image, caption=f"Imagen generada: {prompt_image}", use_container_width=True)
                respuesta = f"He generado una imagen para: {prompt_image}"

            elif use_agent:
                if agente == "CodeAgent":
                    respuesta = agentes.agente_programador(user_input)
                elif agente == "WriteAgent":
                    respuesta = agentes.agente_escritor(user_input)
                elif agente == "TranslateAgent":
                    respuesta = agentes.agente_traductor(user_input)
                elif agente == "DataGenAgent":
                    respuesta = agentes.agente_datos(user_input)
            
                # Asegurartse de que el resultado sea una cadena de texto
                if isinstance(respuesta, dict) and "output" in respuesta:
                    respuesta = respuesta["output"]  # Extraer el texto del resultado
                else:
                    respuesta = str(respuesta)  


            # Detectar si el usuario pidió un gráfico y especificó datos
            elif "gráfico" in user_input.lower():
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
            
            # Preguntas sobre las tareas
            elif "tareas de" in user_input.lower():
                persona = user_input.lower().replace("tareas de", "").strip()
                tareas_asignadas = [task for task in st.session_state["tasks"] if task["assigned_to"].lower() == persona.lower()]
                if tareas_asignadas:
                    respuesta = f"Tareas asignadas a {persona}: " + ", ".join([task["task"] for task in tareas_asignadas])
                else:
                    respuesta = f"No hay tareas asignadas a {persona}."
            elif "tareas activas" in user_input.lower():
                tareas_activas = [task for task in st.session_state["tasks"] if not task["completed"]]
                if tareas_activas:
                    respuesta = "Tareas activas: " + ", ".join([task["task"] for task in tareas_activas])
                else:
                    respuesta = "No hay tareas activas."
            else:
                respuesta = cadena.invoke({"user_input": user_input, "chat_history": st.session_state["chat_history"]})

            # Agregar al historial
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=respuesta))

            # Mostrar historial de chat
            st.markdown("### Chat")
            for mensaje in st.session_state["chat_history"]:
                role = "Usuario" if isinstance(mensaje, HumanMessage) else bot_name
                with st.chat_message(role):
                    st.write(mensaje.content) 

            # Elimina el texto pdf y la imagen del array de chat_history para no interferir con preguntas siguientes
            for i in range(len(st.session_state["chat_history"])):
                if "Imagen:" in st.session_state["chat_history"][i].content:
                        st.session_state["chat_history"].pop(i)
                        break
            for i in range(len(st.session_state["chat_history"])):
                if "Texto:" in st.session_state["chat_history"][i].content:
                        st.session_state["chat_history"].pop(i)
                        break       

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
