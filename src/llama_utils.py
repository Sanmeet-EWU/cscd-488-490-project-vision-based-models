from groq import Groq
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from src.keys import load_api_key
from sentence_transformers import SentenceTransformer, util
import numpy as np

load_api_key("GROQ_API_KEY")
load_api_key("OPENAI_API_KEY")

client = Groq()
MODEL = "llama-3.1-8b-instant"
PERSIST_DIR = "./rag/vector_storage"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

# Load embedding model for similarity checking
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the sentence transformer model
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_user_input(user_prompt, language, selected_crop=None):
    """
    Processes user input by retrieving relevant knowledge and generating a response.
    If no crop is selected, retrieve information from all crops.
    """
    user_message = {"role": "user", "content": user_prompt}
    st.session_state["current_chat_history"].append(user_message)
    
    display_current_chat()

    # Ensure retrieved_docs is iterable
    query_engine = index.as_query_engine()
    retrieved_docs = query_engine.query(user_prompt)

    if hasattr(retrieved_docs, "documents"):  
        retrieved_docs = retrieved_docs.documents  # Handle object response
    elif not isinstance(retrieved_docs, list):  
        retrieved_docs = [retrieved_docs]  # Wrap single object in list

    # Extract text & filter by crop
    retrieved_chunks = []
    for doc in retrieved_docs:
        doc_text = doc.get_text() if hasattr(doc, 'get_text') else str(doc)
        doc_crop = doc.metadata.get("crop", "Unknown")  # Handle missing metadata safely

        # Filter if a crop is selected, otherwise retrieve all docs
        if selected_crop and selected_crop != "Select a Crop":
            if doc_crop == selected_crop:
                retrieved_chunks.append(doc_text)
        else:
            retrieved_chunks.append(doc_text)  # No crop selected, include all

    # Ensure at least some context is retrieved
    if not retrieved_chunks:
        assistant_response = "I am Sorry, I am a farmer assistant and I do not have information on this topic."
        st.chat_message("assistant").markdown(assistant_response)
        st.session_state["current_chat_history"].append({"role": "assistant", "content": assistant_response})
        return user_message, {"role": "assistant", "content": assistant_response}

    context_text = "\n".join(retrieved_chunks)

    # Compute semantic similarity between user query & retrieved context
    query_embedding = similarity_model.encode(user_prompt, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(query_embedding, context_embedding).item()

    similarity_threshold = 0.5  # Adjust based on testing
    print(f"Similarity Score: {similarity_score}")
    # Reject queries that are outside the domain
    if similarity_score < similarity_threshold:
        assistant_response = "I'm sorry, but your question seems unrelated to the available knowledge on crops."
        st.chat_message("assistant").markdown(assistant_response)
        st.session_state["current_chat_history"].append({"role": "assistant", "content": assistant_response})
        print(f"⚠️ Low similarity score ({similarity_score}): Query rejected.")
        return user_message, {"role": "assistant", "content": assistant_response}
    
    # Construct message history
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

def generate_clip_description(caption, language):   
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
