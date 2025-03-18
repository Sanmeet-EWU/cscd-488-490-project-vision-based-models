import streamlit as st
st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

# Importing CLIP and LLaMA utilities
from src.clip_utils import load_custom_clip_model, get_text_features, get_image_features, classify_image, compute_similarity, load_basic_clip_model
from src.classes import get_classes 
from src.llama_utils import generate_clip_description, process_user_input, display_current_chat, generate_chat_title
from src.oauth import get_login_url, get_google_info
from src.firebase_config import create_user_if_not_exists, create_new_chat, fetch_chat_history, load_chat, update_chat_history, add_feedback

def main():
    # Ensure session state is initialized
    if "current_chat_history" not in st.session_state:
        st.session_state["current_chat_history"] = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    previous_chat_recieved = False
    # === SIDEBAR: Google Sign-In & Chat History ===
    with st.sidebar:
        st.title("Account")
        user = None

        language = st.selectbox("Select Language", ["English", "Español", "French"])
        
        if "google_user" not in st.session_state:
            login_url = get_login_url()
            st.markdown(f'<a href="{login_url}" target="_self"><button style="width: 100%;">🔑 Sign in with Google</button></a>', unsafe_allow_html=True)

            query_params = st.query_params
            if "code" in query_params:
                auth_code = query_params["code"]
                user_info = get_google_info(auth_code)
                if user_info:
                    user_id = user_info["sub"]
                    create_user_if_not_exists(user_id, user_info["name"], user_info["email"])
                    user_info["chat_history"] = fetch_chat_history(user_id)
                    st.session_state["google_user"] = user_info
                    st.rerun()
        else:
            st.markdown(
                  f"""
                  <style>
                  div[data-testid="stButton"] > button {{
                      width: 100%;
                      display: block;
                      overflow: hidden;
                      white-space: nowrap;
                      /* padding: 12px 16px; */
                      text-overflow: ellipsis;
                  }}
                  </style>
                  """,
                  unsafe_allow_html=True
             )
            
            user = st.session_state["google_user"]
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(user.get("picture", "user.png"), width=40)
            with col2:
                st.write(f"**{user['name']}**")
                st.write(f"📧 {user['email']}")

            st.subheader("💬 Feedback")
            feedback = st.text_area("Let us know your thoughts!", key="feedback_input")

            if st.button("Submit Feedback"):
                add_feedback(user["sub"], feedback)
                st.success("Thank you for your feedback!")

            if st.button("Logout", use_container_width=True):
                st.session_state.pop("google_user", None)
                st.rerun()

        st.divider()

        if user:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state["current_chat_history"] = []
                st.session_state.pop("current_chat_id", None)
                st.rerun()

            st.subheader("📝 Chat History")
            for chat in user["chat_history"]:
                chat_title = chat["title"]
                chat_id = chat["chat_id"]
                if st.button(f"{chat_title}", key=chat_id):
                    load_chat(user["sub"], chat_id)
                    st.session_state["current_chat_id"] = chat_id
                    previous_chat_recieved = True

    # ==== MAIN CONTENT ====
    st.title("🌱 CLIP Crop & Disease Detection")
    st.write("Upload an image and let AI detect potential crop diseases and provide insights.")

    selected_crop = st.selectbox("Select crop type for better accuracy", ["Select a Crop", "Wheat", "Rice", "Corn", "Potato"], 
        key="crop_selection_popup", help="Select a crop or use all available classes.")
    
    custom_captions = get_classes(selected_crop)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="unique_key_1")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Prevent reprocessing if the same image is uploaded
        if uploaded_file != st.session_state.uploaded_image:
            st.session_state.uploaded_image = uploaded_file

            # Load CLIP model (cached)
            model, preprocess, device = load_basic_clip_model()

            st.image(image, caption="Uploaded Image", width=400)

            # Perform classification
            best_caption, confidence = classify_image(image, model, preprocess, device, custom_captions)

            # If chat history is empty, generate response
            if not st.session_state["current_chat_history"]:
                clip_description = generate_clip_description(best_caption, language)
                
                if user is not None:
                    if "current_chat_id" not in st.session_state:
                        title = generate_chat_title(clip_description, language)
                        st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
                    update_chat_history(user["sub"], st.session_state["current_chat_id"], clip_description)

    # Chat Interaction
    st.subheader("💬 Chat with LLAMA")
    if previous_chat_recieved:
        display_current_chat()
        
    user_prompt = st.chat_input("Ask LLAMA about this diagnosis...")
    if user_prompt:
        messages = process_user_input(user_prompt, language)  # Handle user query
        if user:
            if "current_chat_id" not in st.session_state:
                title = generate_chat_title(user_prompt, language)
                st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
            update_chat_history(user["sub"], st.session_state["current_chat_id"], messages[0])
            update_chat_history(user["sub"], st.session_state["current_chat_id"], messages[1])

if __name__ == "__main__":
    main()
