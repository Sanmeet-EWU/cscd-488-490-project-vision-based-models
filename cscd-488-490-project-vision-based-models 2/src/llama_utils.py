from groq import Groq
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from src.keys import load_api_key

load_api_key("GROQ_API_KEY")
load_api_key("OPENAI_API_KEY")

client = Groq()
MODEL = "llama-3.1-8b-instant"
PERSIST_DIR = "./rag/vector_storage"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)


def process_user_input(user_prompt, language):
    user_message = {"role": "user", "content": user_prompt}
    st.session_state["current_chat_history"].append(user_message)
    
    display_current_chat()

    query_engine = index.as_query_engine()
    retrieved_docs = query_engine.query(user_prompt)  
    context_text = "\n".join([doc.get_text() for doc in retrieved_docs]) if hasattr(retrieved_docs, 'get_text') else str(retrieved_docs)

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use retrieved knowledge to answer the user's question in {language}."},
        {"role": "system", "content": f"Relevant Context:\n{context_text}"},
        *st.session_state["current_chat_history"],
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    assistant_message = {"role": "assistant", "content": assistant_response}
    st.session_state["current_chat_history"].append(assistant_message)
    st.chat_message("assistant").markdown(assistant_response)

    return user_message, assistant_message


def generate_clip_description(caption, confidence, language):   
    prompt = (
        f"You have been provided a picture of a {caption}."
        f"You should say what it is, and be open to answering questions about it."
        f"Avoid mentioning that you have been provided a description"
    )

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Provide the information in {language} language."},
        {"role": "user", "content": prompt}, 
    ]

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    formatted_response = {"role": "assistant", "content": assistant_response}
    
    st.session_state["current_chat_history"].append(formatted_response)
    display_current_chat()
    
    return formatted_response  # Ensure it returns the formatted response

def display_current_chat():
    for message in st.session_state["current_chat_history"]:
        st.chat_message(message["role"]).markdown(message["content"])

def generate_chat_title(user_prompt, language):
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Generate a concise 3-4 word title for the following conversation in {language} language."},
        {"role": "user", "content": f"Content: {user_prompt}\n\nTitle:"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=10, 
        n=1,
        stop=["\n"]
    )
    
    return response.choices[0].message.content.strip()
