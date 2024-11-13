import ollama
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st


llm = OllamaLLM(model="llama3.2:1b", temperature=0.2)

def main():

    # Título y Descripción de la Aplicación
    st.set_page_config(page_title="wellAI", layout="wide")
    st.image("WellAI_logo.png", width=150)

    #Configurar chatbot
    bot_name = "WellAI"
    bot_prompt = f"Eres una IA desarrollada por iewell y te llamas {bot_name} y respondes preguntas, creas código, gráficos y culquier cosa que te pregunte el usuario"

    #Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    #Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", bot_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_input}")
    ])

    cadena = prompt_template | llm
    user_input = st.text_input("Escribe tu pregunta", key="user_input")

    # Sección de Contacto o Ayuda
    st.sidebar.title("wellAI - Menú")
    st.sidebar.write("Explora las funcionalidades de wellAI o contacta con soporte.")

    st.sidebar.header("Acerca de wellAI")
    st.sidebar.write("wellAI es una IA desarrollada por iewell company.")

    st.sidebar.header("Contacto")
    st.sidebar.write("Para más información, contáctanos en contacto@iewell.com")

    if st.button("Preguntar"):
        if user_input.lower() == "adios":
            st.stop()
        else:
            respuesta = cadena.invoke({"user_input": user_input, "chat_history": st.session_state["chat_history"]})
            #Agregamos al historial
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=respuesta))

    chat_res = ""
    for mensaje in st.session_state["chat_history"]:
        if isinstance(mensaje, HumanMessage):
            chat_res += f"Usuario: {mensaje.content}\n"
        elif isinstance(mensaje, AIMessage):
            chat_res += f"{bot_name}: {mensaje.content}\n"

    st.text_area("Chat", value=chat_res, height=300, key="res_area")        


if __name__ == "__main__":
    main()    