import gradio as gr
from PIL import Image
import os, random
from src.predict import predict_breed
from src.breed_info import get_breed_info
from src.similarity import get_similar_breeds
import base64

size_scales = {
    "Extra Small": 40,
    "Small": 60,
    "Medium": 80,
    "Large": 100,
    "Extra Large": 120
}

def render_size_silhouettes(breed_size):
    size_order = ["Extra Small", "Small", "Medium", "Large", "Extra Large"]
    base_img_path = "data/size.png"

    # Encode once
    with open(base_img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{encoded}"

    html = '<div style="display:flex; gap:10px; align-items:flex-end;">'
    for size in size_order:
        height = size_scales[size]
        color_filter = "opacity:0.2;"  # dim by default
        if size in breed_size:
            color_filter = "opacity:1; filter: drop-shadow(0 0 5px blue);"
        html += f'<img src="{data_uri}" style="height:{height}px; {color_filter}"/>'
    html += '</div>'
    return html

def predict_and_info(img):
    pred, top3 = predict_breed(img)
    breed = pred
    info = get_breed_info(breed)

    # --- Text info ---
    result_text = f"## {info['breed']}\n"
    if info.get("alternateName"):
        result_text += f"**Also known as:** {info['alternateName']}\n\n"

    if info.get("countryOfOrigin"):
        result_text += f"**Country of origin:**\n\n"
        for country, flag_path in info["countryOfOrigin"].items():
            if os.path.exists(flag_path):
                with open(flag_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/svg+xml" if flag_path.lower().endswith(".svg") else "image/png"
                data_uri = f"data:{mime};base64,{encoded}"
                result_text += f'<img src="{data_uri}" alt="{country} flag" width="30"/> {country}<br>\n'
            else:
                result_text += f"{country} (FLAG NOT FOUND)<br>\n"

    if info.get("size"):
        result_text += f"\n**Size:** {render_size_silhouettes(info['size'])}\n\n"
    
    if info.get("top_attributes"):
        result_text += "**Best known for:**\n"
        for attr, val in info["top_attributes"]:
            result_text += f"- {attr}\n"

    # --- Gallery for predicted breed ---
    main_gallery = []
    img_dir = os.path.join("data", "images", breed)
    if os.path.exists(img_dir):
        all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        main_gallery = random.sample(all_imgs, min(10, len(all_imgs)))

    # --- Gallery for similar breeds ---
    similar_gallery = []
    similar = get_similar_breeds(info['breed'])
    for s in similar:
        sim_dir = os.path.join("data", "images", s)
        if os.path.exists(sim_dir):
            sim_imgs = [os.path.join(sim_dir, f) for f in os.listdir(sim_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if sim_imgs:
                # pick one random image and label it with breed name
                img_path = random.choice(sim_imgs)
                similar_gallery.append([img_path, s])  # [image, label] for gallery

    # Generate size silhouette visualization
    size_html = render_size_silhouettes(info.get("size", []))

    return result_text, main_gallery, similar_gallery, size_html

# --- Gradio Interface ---
with gr.Blocks(css=".gallery {max-height: 250px; overflow-y: auto;}") as demo:
    gr.Markdown("# Dog Breed Classifier")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload a Dog Image")
            btn = gr.Button("Predict")
        with gr.Column(scale=2):
            result = gr.Markdown()
            main_gallery = gr.Gallery(label="Examples of predicted breed", show_label=True,
                                      elem_classes="gallery", columns=5, rows=2)
            similar_gallery = gr.Gallery(label="Similar breeds", show_label=True,
                                         elem_classes="gallery", columns=3, rows=1)
            size_display = gr.HTML(label="Size Indicator")


    btn.click(fn=predict_and_info,
              inputs=img_input,
              outputs=[result, main_gallery, similar_gallery])

if __name__ == "__main__":
    demo.launch()
