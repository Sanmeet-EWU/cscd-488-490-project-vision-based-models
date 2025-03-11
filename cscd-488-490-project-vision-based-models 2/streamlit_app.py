import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

# Importing CLIP and LLaMA utilities
from src.clip_utils import load_custom_clip_model, get_text_features, get_image_features, compute_similarity, \
    load_clip_model

from src.classes import get_candidate_captions
from src.oauth import get_login_url, get_google_info
from src.firebase_config import create_user_if_not_exists, create_new_chat, fetch_chat_history, load_chat, update_chat_history
from src.llama_utils import generate_clip_description, process_user_input, display_current_chat, generate_chat_title

def main():
    st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")

    # Load the fine-tuned CLIP model
    model, preprocess, device = load_clip_model()  # Load the default CLIP model instead

    candidate_captions = get_candidate_captions()
    clicked_previous_chat = False  # temporary
    prompts = 0  # Track # of prompts

    # Initialize session state
    if "current_chat_history" not in st.session_state:
        st.session_state["current_chat_history"] = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # ==== SIDEBAR (Chat History + Google Sign-In) ====
    with st.sidebar:
        st.title("Account")
        user = None

        language = st.selectbox("Select Language", ["English", "Spanish"])

      


        # Google Sign-In Section
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
            user = st.session_state["google_user"]
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(user["picture"], width=40)
            with col2:
                st.write(f"**{user['name']}**")
                st.write(f"📧 {user['email']}")

            if st.button("Logout", use_container_width=True):
                st.session_state.pop("google_user", None)
                st.rerun()

        st.divider()

        if "google_user" in st.session_state:
            user_info = st.session_state["google_user"]

            if st.button("➕ New Chat", use_container_width=True):
                st.session_state["current_chat_history"] = []
                st.session_state.pop("current_chat_id", None)
                st.rerun()

            st.subheader("📝 Chat History")
            for chat in user_info["chat_history"]:
                chat_title = chat["title"]
                chat_id = chat["chat_id"]
                if st.button(f"{chat_title}", key=chat_id):
                    load_chat(user_info["sub"], chat_id)
                    clicked_previous_chat = True
                    st.session_state["current_chat_id"] = chat_id

    # ==== MAIN CONTENT ====
    st.title("🌱 CLIP Crop & Disease Detection")
    st.write("Upload an image and let AI detect potential crop diseases and provide insights.")

    # Image Upload & Processing
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        # Extract features and compute similarities
        text_features = get_text_features(candidate_captions, device, model)

        # Apply preprocessing transformations
        transformed_image = preprocess(image)  # Convert to tensor [3, 224, 224]
        transformed_image = transformed_image.unsqueeze(0).to(device)  # Add batch dimension [1, 3, 224, 224]

        # 🔥 Pass original image (PIL format) to get_image_features
        image_features = get_image_features(image, device, model, preprocess)


        # 🔥 Use `classify_image()` for consistency
        #logits, confidences = classify_image(image_features, model)

        # Select top N predictions
        N = 3  # Change as needed
        #top_indices = torch.argsort(confidences, descending=True)[:N]  # Get top N indices

        # ✅ Convert tensor indices to list of integers before using as list indices
        #top_classes = [candidate_captions[idx.item()] for idx in top_indices]
        #top_confidences = [confidences[idx].item() * 100 for idx in top_indices]
        ''' 
        # Display predictions in Streamlit
        st.write("🔍 **Top Predictions:**")
        for i in range(N):
            st.write(f"**{i+1}. {top_classes[i]}** - {top_confidences[i]:.2f}% confidence")


        # Print debugging info
        st.write("🔍 Model Predictions (Classifier-Based):")
        st.write(f"Logits: {logits.tolist()}")
        st.write(f"Confidence Scores: {top_confidences}")

        top_captions = [candidate_captions[idx.item()] for idx in top_indices]  # Convert tensor index to int


        # Display top prediction
        best_caption = top_captions[0]
        confidence = top_confidences[0]
        st.success(f"🔍 Prediction: **{best_caption}** with {confidence:.2f}% confidence")



        # Generate AI response if no chat history exists
        if not st.session_state["current_chat_history"]:
            clip_description = generate_clip_description(best_caption, confidence, language)
            if user is not None:
                if "current_chat_id" not in st.session_state:
                    title = generate_chat_title(clip_description)
                    st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
                update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], clip_description)
'''
    # Chat Interaction
    st.subheader("💬 Chat with LLAMA")
    if clicked_previous_chat:
        display_current_chat()

    user_prompt = st.chat_input("Ask LLAMA about this diagnosis...")
    if user_prompt:
        messages = process_user_input(user_prompt, language)  # Handle user query
        if user is not None:
            if "current_chat_id" not in st.session_state:
                title = generate_chat_title(user_prompt)
                st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[0])
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[1])
        prompts += 1

if __name__ == "__main__":
    main()
