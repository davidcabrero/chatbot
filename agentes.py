from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
import ollama

llm = LLM(
        model="ollama/llama3.2:1b",
        temperature = 0.5,
        base_url="http://127.0.0.1:11434",
    )

# Agente buscador en web
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
        description=
            "Corriges los errores ortográficos y mejoras el léxico para que sea el apropiado. \n"
            "Redactas correctamente la respuesta detalladamente\n",
        expected_output="Respuesta a la pregunta detallada y corregida",
        agent=web_scraper
    )

    crew = Crew(
        agents=[web_scraper],
        tasks=[scraper],
        verbose=True
    )

    result = crew.kickoff(inputs={"pregunta": pregunta, "contexto": texto})
    return result

#Agente Programador
def agente_programador(pregunta):
    # Definir el agente programador
    programador = Agent(
        role="Experto en programación",
        goal="Resuelves este problema de programación: {pregunta} con código y escribes código de clalidad, dando al usuario la respuesta con código en cualquier lenguaje.",
        backstory="Eres un agente experto en programación que escribes código en todos los lenguajes.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # Definir la tarea
    tarea = Task(
        description=
            "Revisas el código para que sea lo más limpio posible. \n"
            "Aseguras que el formato del código sea correcto y se vea claro\n",
        expected_output="Código del programa correcto.",
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

# Agente Uso internet en Chatbot
def agente_internet(pregunta):
    result = None
    return result

