from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import FileWriterTool
import ollama
import csv
import io
from flask import send_file
from googlesearch import search

# Para ejecutar los agentes con el modelo local
llm_local = LLM(
        model="ollama/llama3.2:1b",
        temperature = 0.5,
        base_url="http://127.0.0.1:11434",
    )
# Para ejecutar los agentes con la api de groq
llm = LLM(
    model= "groq/gemma2-9b-it",
    temperature = 0.5,
    base_url="https://api.groq.com/openai/v1",
        api_key="API_KEY"
)

# Agente de web scraping
def web_scraping_tool(pregunta, url):
    tool = ScrapeWebsiteTool(url)
    texto = tool.run()

    # Definir los agentes
    web_scraper = Agent(
        role="Experto investigador",
        goal="Resuelve la pregunta {pregunta} utilizando la información de {contexto}.",
        backstory="Eres un agente que resuelve cualquier pregunta con la info de {contexto}.",
        allow_delegation=False,
        verbose=True,
        tools=[tool],
        llm=llm,
    )

    scraper = Task(
        description=(
            "Corriges los errores ortográficos y mejoras el léxico para que sea el apropiado. \n"
            "Redactas correctamente la respuesta detalladamente\n"
        ),
        expected_output="Respuesta corregida gramaticalmente y ordenada",
        agent=web_scraper
    )

    crew = Crew(
        agents=[web_scraper],
        tasks=[scraper],
        verbose=True
    )

    # Ejecutar la tarea
    result = crew.kickoff(inputs={"pregunta": pregunta, "contexto": texto})
    return result

#Agente Programador
def agente_programador(pregunta):
    # Definir el agente programador
    programador = Agent(
        role="Experto en programación",
        goal="Haces con código el programa que pide el usuario: {pregunta}. Escribes todo el código.",
        backstory="Eres un agente experto en programación que escribes código en todos los lenguajes.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Revisas el código para que sea lo más limpio posible.",
        expected_output="Código del programa.",
        agent=programador
    )

    crew = Crew(
        agents=[programador],
        tasks=[tarea],
        verbose=True
    )

    result = crew.kickoff(inputs={"pregunta": pregunta})
    return result

# Agente Escritor
def agente_escritor(pregunta):
    # Definir el agente escritor
    escritor = Agent(
        role="Experto escritor",
        goal="Escribes un texto de calidad sobre {pregunta}.",
        backstory="Eres un agente experto en escritura que redactas textos e informes de calidad.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Corriges los errores ortográficos y mejoras el léxico para que sea el apropiado. \n"
            "Redactas correctamente la respuesta detalladamente\n",
        expected_output="Respuesta a la pregunta detallada y corregida",
        agent=escritor
    )

    crew = Crew(
        agents=[escritor],
        tasks=[tarea],
        verbose=True
    )

    result = crew.kickoff(inputs={"pregunta": pregunta})
    return result

# Agente traductor

def agente_traductor(pregunta):
    # Definir el agente traductor
    traductor = Agent(
        role="Experto traductor",
        goal="Traduces este tecto: {pregunta} en el idioma {idioma}.",
        backstory="Eres un agente que traduce con calidad al {idioma}.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Corriges los errores ortográficos y mejoras el léxico para que sea el apropiado en {idioma}. \n"
            "Redactas correctamente la respuesta detalladamente con el lenguaje más natural en {idioma}\n",
        expected_output="Texto traducido natural y corregido",
        agent=traductor
    )

    crew = Crew(
        agents=[traductor],
        tasks=[tarea],
        verbose=True
    )

    result = crew.kickoff(inputs={"pregunta": pregunta})
    return result

# Agente para generar Datos
def agente_datos(pregunta):

    # Definir el agente generador de datos
    generador_datos = Agent(
        role="Experto generador de datos",
        goal="Generas datos sintéticos para {pregunta} realistas para un csv."
             "Debes dar al menos 50 filas de datos sintéticos y mínimo 5 columnas."
             "Utilizas markdown para dar formato a los datos y que se vean claros.",
        backstory="Eres un agente que genera datos sintéticos aleatorios sobre cualquier tema.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Revisas los datos y aseguras que se dan al menos 50 filas de datos con un formato markdown claro.\n",
        expected_output="Datos generados",
        agent=generador_datos
    )

    crew = Crew(
        agents=[generador_datos],
        tasks=[tarea],
        verbose=True
    )

    # Ejecutar la tarea
    result = crew.kickoff(inputs={"pregunta": pregunta})

    return result

def agente_extraccion_documentos(pregunta, contenido):
    
    # Definir el agente
    extractor = Agent(
        role="Especialista en extracción de información",
        goal="Extrae información clave de documentos y responde la pregunta: {pregunta}.",
        backstory="Eres un agente experto en analizar documentos y responder preguntas con base en su contenido: {contenido}.",
        llm=llm,
    )
    
    # Definir la tarea
    tarea = Task(
        description="Analiza el documento y responde la pregunta proporcionada.",
        expected_output="Respuesta basada en el contenido del documento.",
        agent=extractor
    )
    
    # Crear el crew y ejecutar la tarea
    crew = Crew(agents=[extractor], tasks=[tarea])
    result = crew.kickoff(inputs={"pregunta": pregunta, "contenido": contenido})  # Asegúrate de incluir 'contenido'
    return result

def agente_internet(pregunta):
    try:
        resultados = []
        for resultado in search(pregunta, num_results=5):
            resultados.append(resultado)
        
        texto_resultados = "\n".join(resultados) if resultados else "No se encontraron resultados relevantes."
    
    except Exception as e:
        texto_resultados = f"Error en la búsqueda: {str(e)}"
    
    investigador = Agent(
        role="Investigador en línea",
        goal="Busca información en internet y responde la pregunta: {pregunta}. Los resultados de internet son: {contexto}.",
        backstory="Eres un experto en búsqueda en línea y extraes la mejor información de la web.",
        llm=llm,
    )
    
    tarea = Task(
        description="Busca en internet y proporciona una respuesta basada en los mejores resultados encontrados.",
        expected_output="Resumen de la información encontrada en internet.",
        agent=investigador
    )
    
    crew = Crew(agents=[investigador], tasks=[tarea])
    result = crew.kickoff(inputs={"pregunta": pregunta, "contexto": texto_resultados})
    return result

def agente_matematico(pregunta):
    # Definir el agente matemático
    matematico = Agent(
        role="Experto matemático",
        goal="Resuelves problemas matemáticos y explicas los pasos: {pregunta}.",
        backstory="Eres un agente experto en matemáticas que resuelve problemas y explica los pasos.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Revisas los cálculos y explicas cada paso de forma clara.",
        expected_output="Respuesta detallada con pasos explicativos.",
        agent=matematico
    )

    crew = Crew(
        agents=[matematico],
        tasks=[tarea],
        verbose=True
    )

    result = crew.kickoff(inputs={"pregunta": pregunta})
    return result

