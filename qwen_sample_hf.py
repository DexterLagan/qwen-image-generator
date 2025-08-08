from diffusers import DiffusionPipeline
import torch
import argparse

MODEL_VARIANTS = {
    "fp8": "Qwen/Qwen-Image",
    "fp16": "Qwen/Qwen-Image-FP16",
}

parser = argparse.ArgumentParser(description="Sample Qwen-Image generation script.")
parser.add_argument(
    "--variant",
    choices=MODEL_VARIANTS.keys(),
    default="fp8",
    help="Model precision to use",
)
args = parser.parse_args()

model_name = MODEL_VARIANTS[args.variant]

if torch.cuda.is_available():
    torch_dtype = torch.float16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
try:
    if device == "cuda":
        pipe.to(device)
    else:
        pipe.to("cpu")
except RuntimeError as e:
    if "out of memory" in str(e).lower() and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        raise

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图",
}

prompt = (
    "A coffee shop entrance features a chalkboard sign reading \"Qwen Coffee 😊 $2 per cup,\" "
    "with a neon light beside it displaying \"通义千问\". Next to it hangs a poster showing a "
    "beautiful Chinese woman, and beneath the poster is written \"π≈3.1415926-53589793-23846264-33832795-02384197\". "
    "Ultra HD, 4K, cinematic composition"
)
negative_prompt = " "  # using an empty string if no concept to remove

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

image.save("example.png")

