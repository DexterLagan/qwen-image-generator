import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from datetime import datetime
from pathlib import Path
import random

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Image", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen-Image", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Output folder
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Resolution and aspect ratio mappings
aspect_ratios = {
    "1:1 (square)": (1, 1),
    "4:3 (classic)": (4, 3),
    "3:2 (photo)": (3, 2),
    "16:9 (HD widescreen)": (16, 9),
    "2:3 (portrait)": (2, 3),
}
resolutions = {
    "512": 512,
    "768": 768,
    "1024": 1024
}

def generate_image(prompt, negative_prompt, resolution, aspect_ratio, manual_width, manual_height,
                   steps, guidance_scale, seed, randomize_seed):
    if not prompt.strip():
        return None

    # Adjust seed
    if randomize_seed:
        seed = random.randint(0, 999999)

    # Compute resolution
    if manual_width > 0 and manual_height > 0:
        width, height = manual_width, manual_height
    else:
        base_res = resolutions[resolution]
        ratio_w, ratio_h = aspect_ratios[aspect_ratio]
        gcd = torch.gcd(torch.tensor(ratio_w), torch.tensor(ratio_h)).item()
        width = int(base_res * (ratio_w // gcd))
        height = int(base_res * (ratio_h // gcd))

    # Prepare input
    inputs = processor(
        text=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated = model.generate(**inputs)

    image = processor.decode(generated[0])

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"qwen_image_{timestamp}_seed{seed}.png"
    image.save(filename)

    return image

# Gradio UI
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe the image you want..."),
        gr.Textbox(label="Negative Prompt", placeholder="Optional: things you don't want in the image"),
        gr.Dropdown(label="Base Resolution", choices=list(resolutions.keys()), value="1024"),
        gr.Dropdown(label="Aspect Ratio", choices=list(aspect_ratios.keys()), value="1:1 (square)"),
        gr.Number(label="Manual Width (override)", value=0, precision=0),
        gr.Number(label="Manual Height (override)", value=0, precision=0),
        gr.Slider(label="Steps", minimum=10, maximum=100, step=1, value=50),
        gr.Slider(label="CFG (Guidance Scale)", minimum=1.0, maximum=20.0, step=0.1, value=7.5),
        gr.Slider(label="Seed", minimum=0, maximum=999999, step=1, value=random.randint(0, 999999)),
        gr.Checkbox(label="Randomize Seed", value=True),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Qwen-Image Diffusion GUI",
    description="Text-to-Image with manual or automatic resolution, seed randomization, and negative prompt support."
)

interface.launch()
