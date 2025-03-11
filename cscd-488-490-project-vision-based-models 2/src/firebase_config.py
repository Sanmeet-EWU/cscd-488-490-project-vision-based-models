import firebase_admin
from firebase_admin import credentials, firestore
import os
import uuid  # For generating unique chat IDs
from datetime import datetime, timezone
import streamlit as st
# Check if Firebase is already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
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
    