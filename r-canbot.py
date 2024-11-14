from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Initialize model and prompt outside of function to avoid re-initialization
template = """
Answer the question below.
Conversation history: {context}
Question: {question}
Answer: 
"""

model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    # Load or initialize conversation context from session state
    if "context" not in st.session_state:
        st.session_state.context = ""
    
    # User input field
    user_input = st.text_input("Type your question here: ")

    if st.button("Get Answer"):
        if user_input.lower() == "exit":
            st.write("Ending conversation.")
            return  # Exit the function
        
        if user_input:
            result = chain.invoke({"context": st.session_state.context, "question": user_input})
            st.write("Bot:", result)
            st.session_state.context += f"\nUser: {user_input}\nAI: {result}"
        else:
            st.write("Bot: You haven't entered anything.")

if __name__ == "__main__":
    # Streamlit UI
    st.title("Build your own LLM model with llama3.1")
    handle_conversation()