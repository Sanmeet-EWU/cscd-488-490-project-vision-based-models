classes = {
    "Healthy Wheat": "Green wheat plant with smooth leaves. No spots, lesions, or yellowing.",
    "Wheat Brown Rust": "Wheat leaves with round, reddish-brown spots and pustules.",
    "Wheat Yellow Rust": "Wheat leaves with long yellow streaks and bright yellow pustules.",
    "Healthy Rice": "Green rice leaf with no spots, streaks, or damage.",
    "Rice Brown Spot": "Rice leaf with brown circular spots, dark edges, and gray centers.",
    "Rice Leaf Blast": "Rice leaf with long, narrow lesions with dark brown edges and pale centers.",
    "Rice Neck Blast": "Dark lesions on rice panicle neck, leading to weak stems.",
    "Potato Early Blight": "Potato leaf with dark brown circular spots with yellow halos.",
    "Potato Late Blight": "Potato leaf with dark, irregular lesions.",
    "Corn Grey Leaf Spot": "Corn leaf with long, gray lesions along veins.",
    "Corn Healthy": "Green corn leaf with a smooth surface, intact edges.",
    "Corn Northern Leaf Blight": "Corn leaf with long, grayish-brown elliptical lesions.",
    "Corn Common Rust": "Corn leaf with raised, reddish-brown pustules.",
}

def get_candidate_captions():
    return [f"{cls}: {desc}" for cls, desc in classes.items()]
