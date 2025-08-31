import gradio as gr
from PIL import Image
import os, random
from src.predict import predict_breed
from src.breed_info import get_breed_info
import base64

def predict_and_info(img):
    pred, top3 = predict_breed(img)
    breed = pred
    info = get_breed_info(breed)

    result_text = f"## {info['breed']}\n"
    if info.get("alternateName"):
        result_text += f"**Also known as:** {info['alternateName']}\n\n"

    # Country of origin with flags
    if info.get("countryOfOrigin"):
        result_text += f"**Country of origin:**\n\n"
        for country, flag_path in info["countryOfOrigin"].items():
            print(f"DEBUG: Country: {country}")
            print(f"DEBUG: Flag path: {flag_path}")
            print(f"DEBUG: Exists? {os.path.exists(flag_path)}")

            if os.path.exists(flag_path):
                with open(flag_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                if flag_path.lower().endswith(".svg"):
                    mime = "image/svg+xml"
                else:
                    mime = "image/png"
                data_uri = f"data:{mime};base64,{encoded}"
                result_text += f'<img src="{data_uri}" alt="{country} flag" width="30"/> {country}<br>\n'
            else:
                result_text += f"{country} (FLAG NOT FOUND)<br>\n"



    if info.get("size"):
        result_text += f"\n**Size:** {', '.join(info['size'])}\n\n"
    if info.get("top_attributes"):
        result_text += "**Best known for:**\n"
        for attr, val in info["top_attributes"]:
            result_text += f"- {attr}: {val}/5\n"

    img_dir = os.path.join("data", "images", breed)
    gallery_imgs = []
    if os.path.exists(img_dir):
        all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        gallery_imgs = random.sample(all_imgs, min(10, len(all_imgs)))

    return result_text, gallery_imgs

# --- Gradio Interface ---
with gr.Blocks(css=".gallery {max-height: 250px; overflow-y: auto;}") as demo:
    gr.Markdown("# Dog Breed Classifier")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload a Dog Image")
            btn = gr.Button("Predict")
        with gr.Column(scale=2):
            result = gr.Markdown()
            gallery = gr.Gallery(label="Examples", show_label=True, elem_classes="gallery", columns=5, rows=2)

    btn.click(fn=predict_and_info, inputs=img_input, outputs=[result, gallery])

if __name__ == "__main__":
    demo.launch()
