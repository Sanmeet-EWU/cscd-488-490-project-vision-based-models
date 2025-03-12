import streamlit as st
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from src.classes import get_classes
import requests


# __all__ = ["load_custom_clip_model", "get_text_features", "get_image_features", "compute_similarity", "classify_image"]


import torch.nn.functional as F

class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)  # Fine-tuned classifier

    def forward(self, x):
        features = self.model.encode_image(x).float()  # Extract image features
        features = F.normalize(features, dim=-1)  # Normalize features
        return self.classifier(features)  # Predict class logits

    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image):
        return self.model.encode_image(image)

def load_basic_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


MODEL_URL = "https://huggingface.co/rahat15/CLIP-Finetune/resolve/main/clip_finetuned.pth"

MODEL_PATH = "/tmp/clip_finetuned.pth"

def download_model():
    """Ensure the model is downloaded before loading."""
    if not os.path.exists(MODEL_PATH):
        print("🚀 Downloading CLIP model to Streamlit Cloud...")
        with st.spinner("Downloading model... Please wait."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()  # Check for HTTP errors
                
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Model downloaded successfully at {MODEL_PATH}")
            except Exception as e:
                st.error(f"❌ Model download failed: {e}")
                return False
    return True


@st.cache_resource
def load_custom_clip_model(num_classes=13):
    """Load CLIP fine-tuned model after ensuring it's downloaded."""
    model_downloaded = download_model()
    if not model_downloaded:
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        base_model, preprocess = clip.load("ViT-B/32", device=device)
        model = CLIPFineTuner(base_model, num_classes).to(device)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

        # 🔥 Fix: Explicitly set weights_only=False to allow full state_dict loading
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        model.load_state_dict(state_dict)
        model.eval()

        print("✅ CLIP model loaded successfully.")
        return model, preprocess, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Ensure model is loaded
model, preprocess, device = load_custom_clip_model()
if model is None:
    st.stop()  # Stop execution if model fails to load


def get_text_features(captions, device, model):
    text_tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def get_image_features(image, device, model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

def compute_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    
    return similarity

def classify_image(image, model, preprocess, device, custom_captions):
    # Get candidate captions and their features
    text_inputs = torch.cat([clip.tokenize(custom_captions[c]) for c in custom_captions.keys()]).to(device)
    
    image = preprocess(image).unsqueeze(0).to(device)  # Apply CLIP preprocessing

    # Extract image and text features
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity between image and text features
    similarity = image_features @ text_features.T  # Shape: (1, num_classes)

    # Find the caption with the highest similarity score
    predicted_index = similarity.argmax(dim=-1).item()  # Convert tensor to int
    predicted_label = list(custom_captions.keys())[predicted_index]
    best_similarity_score = similarity[0, predicted_index].item()  # Extract top similarity score

    return predicted_label, best_similarity_score  # String, Number
