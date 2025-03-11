import streamlit as st
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

__all__ = ["load_custom_clip_model", "get_text_features", "get_image_features", "compute_similarity", "classify_image"]


import torch.nn.functional as F

class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)  # Fine-tuned classifier

    def forward(self, x):
        features = self.model.encode_image(x).float()  # Extract image features
        features = F.normalize(features, dim=-1)  # ✅ Normalize features
        return self.classifier(features)  # Predict class logits

    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image):
        return self.model.encode_image(image)
    


@st.cache_resource
def load_custom_clip_model(model_path="src/clip_finetuned.pth", classifier_path="src/classifier_weights.pth", num_classes=13):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure paths exist
    model_path = os.path.abspath(model_path)
    classifier_path = os.path.abspath(classifier_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}. Check the path!")

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"❌ Classifier file not found: {classifier_path}. Did you save it?")

    # ✅ Load fine-tuned CLIP backbone
    base_model, _ = clip.load("ViT-B/32", device=device)
    model = CLIPFineTuner(base_model, num_classes).to(device)

    # ✅ Load fine-tuned backbone weights (excluding classifier)
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and "classifier" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # ✅ Debug: Check if classifier weights are actually loaded
    # Debugging - Check classifier weights before and after loading
    print(f"🔍 Before loading, classifier weights sample: {model.classifier.weight[0][:5]}")


    state_dict = torch.load(model_path, map_location=device)

# 🔥 Manually load classifier weights if they exist
    if "classifier.weight" in state_dict and "classifier.bias" in state_dict:
        model.classifier.weight.data = state_dict["classifier.weight"]
        model.classifier.bias.data = state_dict["classifier.bias"]
        st.write("✅ Successfully loaded classifier weights!")

    else:
        st.write("❌ Fine-tuned classifier weights were NOT found in checkpoint!")

    # Load the rest of the model (excluding classifier)
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
   

    classifier_state_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items()}  

    model.classifier.load_state_dict(classifier_state_dict, strict=False)  # Load classifier weights

    print(f"✅ After loading, classifier weights sample: {model.classifier.weight[0][:5]}")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))  
    ])

    return model, preprocess, device



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
    """Compute cosine similarity and return normalized confidence values"""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ text_features.T).squeeze(0)

    # ✅ Scale similarities before softmax to sharpen confidence
    confidences = F.softmax(similarities * 5, dim=0)  # Scale by 5 for better separation
    return similarities, confidences



def classify_image(image_features, model):
    model.eval()

    with torch.no_grad():
        logits = model.classifier(image_features) * 1 

        # 🔥 Apply temperature scaling (Lower temp = stronger predictions)
        temperature = 0.7  # Test values between 0.8 to 1.5
        logits /= temperature  

        probabilities = F.softmax(logits, dim=-1)  

    return logits.squeeze(0), probabilities.squeeze(0)

