from groq import Groq
import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage
from src.keys import load_api_key
from src.firebase_config import update_chat_history

load_api_key("GROQ_API_KEY")
load_api_key("OPENAI_API_KEY")

client = Groq()

MODEL = "llama-3.1-8b-instant"

PERSIST_DIR = "./rag/vector_storage"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

def process_user_input(user_prompt):
    # Append user input to chat history
    user_message = {"role": "user", "content": user_prompt}
    st.session_state["current_chat_history"].append(user_message)

    # Retrieve relevant context from the RAG system
    query_engine = index.as_query_engine()
    retrieved_docs = query_engine.query(user_prompt)  # Fetch relevant data
    context_text = "\n".join([doc.get_text() for doc in retrieved_docs]) if hasattr(retrieved_docs, 'get_text') else str(retrieved_docs)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following retrieved knowledge when answering the user's question."},
        {"role": "system", "content": f"Relevant Context:\n{context_text}"},  # Inject RAG results here
        *st.session_state["current_chat_history"],
    ]

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    assistant_message = {"role": "assistant", "content": assistant_response}
    st.session_state["current_chat_history"].append(assistant_message)

    display_current_chat()

    return user_message, assistant_message

def display_current_chat():
    # Display the entire chat history in correct order
    for message in st.session_state["current_chat_history"]:
        role_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
        st.chat_message(message["role"]).markdown(f"{role_icon} {message['content']}")

# Prompt the LLM and show results to user. Hide initial prompt from user.
def generate_clip_description(caption, confidence):

    prompt = (
        f"You have been provided a picture of a {caption}."
        f"You should say what it is, and be open to answering questions about it."
        f"Avoid mentioning that you have been provided a description"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}, 
    ]

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    st.session_state["current_chat_history"].append({"role": "assistant", "content": assistant_response})


def process_user_input_norag(st, user_prompt):
    st.chat_message("user").markdown(user_prompt)
    st.session_state["current_chat_history"].append({"role": "user", "content": user_prompt})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *st.session_state["current_chat_history"], 
    ]

    # Send the user's message to the LLM and get a response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    st.session_state["current_chat_history"].append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

# Prompt the LLM with a silent instruction (no output)
def process_silent_instruction(hidden_instruction):
    #Sends a silent instruction to the LLM, with no visible input or output to the user.

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": hidden_instruction},
    ]

    # Send the instruction to the LLM (no output displayed)
    client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

def generate_chat_title(user_prompt):
    """Generate a chat title based on content"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Generate a concise and descriptive 2-4 word title for the following content."},
        {"role": "user", "content": f"Content: {user_prompt}\n\nTitle:"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=10, 
        n=1,
        stop=["\n"]
    )
    # Extract title string from the response
    title = response.choices[0].message.content.strip()
    return title