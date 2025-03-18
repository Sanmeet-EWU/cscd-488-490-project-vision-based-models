import firebase_admin
from firebase_admin import credentials, firestore
import os
import uuid  # For generating unique chat IDs
from datetime import datetime, timezone
import streamlit as st
from src.llama_utils import display_current_chat
# Check if Firebase is already initialized
# Load Firebase credentials from secrets.toml
firebase_secrets = st.secrets["firebase"]

# Ensure the private key is correctly formatted with newlines
private_key = firebase_secrets["private_key"].replace('\\n', '\n')

cred = credentials.Certificate({
    "type": firebase_secrets["type"],
    "project_id": firebase_secrets["project_id"],
    "private_key_id": firebase_secrets["private_key_id"],
    "private_key": private_key,
    "client_email": firebase_secrets["client_email"],
    "client_id": firebase_secrets["client_id"],
    "auth_uri": firebase_secrets["auth_uri"],
    "token_uri": firebase_secrets["token_uri"],
    "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": firebase_secrets["client_x509_cert_url"],
})

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

def create_user_if_not_exists(user_id, user_name, user_email):
    """Check if the user exists in Firestore. If not, create a new user document."""
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        user_ref.set({
            "name": user_name,
            "email": user_email,
            "created_at": datetime.now(timezone.utc),
        })

def create_new_chat(user_id, title="New Chat"):
    """Creates a new chat session with placeholder data."""
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)

    chat_ref.set({
        "created_at": datetime.now(timezone.utc),
        "last_updated": datetime.now(timezone.utc),
        "title": title,  # Placeholder title
        "messages": []
    })
    return chat_id

def fetch_chat_history(user_id):
    """Fetches all chat history for the given user (ID and title only)"""
    chat_history = []
    chats_ref = db.collection("users").document(user_id).collection("chats")
    for chat in chats_ref.stream():
        chat_data = chat.to_dict()
        chat_history.append({
            "chat_id": chat.id,
            "title": chat_data.get("title", "Untitled Chat"),
        })
    return chat_history

def update_chat_history(user_id, chat_id, message):
    """Update the chat history with a new message."""
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
    chat_data = chat_ref.get().to_dict()
    chat_data["messages"].append(message)
    chat_data["last_updated"] = datetime.now(timezone.utc)
    chat_ref.set(chat_data)

def load_chat(user_id, chat_id):
    """Load chat history for the selected chat."""
    # Reset current chat history in session state
    st.session_state["current_chat_history"] = []

    # Fetch the chat document from the database
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)
    chat_doc = chat_ref.get()

    if chat_doc.exists:
        chat_data = chat_doc.to_dict()
        messages = chat_data.get("messages", [])
        print(messages)
        st.session_state["current_chat_history"] = messages
    else:
        st.warning("Chat not found.")
    
def add_feedback(user_id, feedback):
    """Add user feedback to the Firestore database."""
    feedback_ref = db.collection("users").document(user_id).collection("feedback").document()
    feedback_ref.set({
        "created_at": datetime.now(timezone.utc),
        "feedback": feedback
    })