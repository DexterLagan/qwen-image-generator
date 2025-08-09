import gradio as gr
from diffusers import DiffusionPipeline
import torch
from datetime import datetime
from pathlib import Path
import os
import sys
import logging
import atexit
import traceback
import glob
import psutil
from huggingface_hub.utils import HfHubHTTPError

# Suppress specific Gradio warnings early
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Note: expandable_segments removed to avoid platform compatibility warnings
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configure paths
MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
LOG_FILE = Path("console.log")

# Supported model variants and their Hugging Face repo IDs
MODEL_VARIANTS = {
    "FP8 (20GB)": {"id": "Qwen/Qwen-Image", "size_gb": 20},
    "FP16 (40GB)": {"id": "Qwen/Qwen-Image-FP16", "size_gb": 40},
}

# Default selection
selected_variant = "FP8 (20GB)"


def set_model_variant(variant: str) -> None:
    """Update global model configuration based on selected variant."""
    global selected_variant, MODEL_NAME, MODEL_SIZE_GB
    selected_variant = variant
    config = MODEL_VARIANTS[variant]
    MODEL_NAME = config["id"]
    MODEL_SIZE_GB = config["size_gb"]


# Initialize globals for default variant
set_model_variant(selected_variant)

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Don't redirect stdout - this breaks Gradio's logging
# Instead, use file logging alongside console logging

# Configure logging properly without interfering with Gradio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')  # File output
    ]
)

logger = logging.getLogger(__name__)

# Global pipeline variable
pipe = None
model_loaded = False

# Register cleanup function
def cleanup():
    """Cleanup function called on exit"""
    try:
        # Clean up logging handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

atexit.register(cleanup)

# Exception handler for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to be handled normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_msg = f"Uncaught exception: {exc_type.__name__}: {exc_value}"
    logger.error(error_msg)
    logger.error("Traceback:")
    logger.error(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    
    # Also print to console in case logging fails
    print(error_msg, file=sys.__stderr__)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.__stderr__)

sys.excepthook = handle_exception

def get_device_and_dtype():
    """Determine optimal device and dtype"""
    # Log system memory info
    memory = psutil.virtual_memory()
    logger.info(f"System RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Attempt small allocation to ensure CUDA is usable
            torch.zeros(1, device="cuda")
            logger.info(f"CUDA device detected: {device_name}")
            logger.info(f"GPU VRAM: {vram_gb:.1f}GB available")
            return "cuda", torch.float16
        except Exception as e:
            logger.warning(f"CUDA not usable ({e}), falling back to CPU")
    else:
        logger.info("No CUDA device detected, using CPU only")

    if memory.available < 8 * (1024**3):  # Less than 8GB available
        logger.warning(f"Low available RAM ({memory.available / (1024**3):.1f}GB). Model loading may be slow or fail.")
    return "cpu", torch.float32

def check_model_files():
    """Check if model files are already downloaded and complete"""
    cache_dir = MODEL_DIR / "cache" / f"models--{MODEL_NAME.replace('/', '--')}"
    logger.info(f"Checking for model files in: {cache_dir}")
    
    if not cache_dir.exists():
        logger.info("Model cache directory not found")
        return {"found": False, "message": "No model cache found", "details": ""}
    
    # Check for key model files (accounting for split files)
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json", 
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors"
    ]
    
    # Check for split model files
    required_patterns = [
        "transformer/diffusion_pytorch_model*.safetensors",
        "text_encoder/model*.safetensors"
    ]
    
    snapshot_dir = cache_dir / "snapshots"
    if not snapshot_dir.exists():
        logger.info("Snapshots directory not found")
        return {"found": False, "message": "Incomplete model cache", "details": "Missing snapshots directory"}
    
    # Find the latest snapshot
    snapshots = list(snapshot_dir.iterdir())
    if not snapshots:
        logger.info("No snapshots found")
        return {"found": False, "message": "Empty model cache", "details": "No model snapshots found"}
    
    latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
    snapshot_id = latest_snapshot.name[:8]  # First 8 chars of commit hash
    logger.info(f"Checking snapshot: {latest_snapshot.name}")
    
    missing_files = []
    found_files = []
    total_size_gb = 0
    
    for file_path in required_files:
        full_path = latest_snapshot / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            found_files.append(file_path)
            try:
                size = full_path.stat().st_size / (1024**3)  # Size in GB
                total_size_gb += size
            except:
                pass
    
    # Check for split model files using glob patterns
    for pattern in required_patterns:
        pattern_path = str(latest_snapshot / pattern)
        matching_files = glob.glob(pattern_path)
        if not matching_files:
            missing_files.append(pattern)
        else:
            found_files.extend([Path(f).relative_to(latest_snapshot) for f in matching_files])
            try:
                for f in matching_files:
                    size = Path(f).stat().st_size / (1024**3)
                    total_size_gb += size
            except:
                pass
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
        return {
            "found": False, 
            "message": f"Incomplete Qwen-Image model (snapshot {snapshot_id})", 
            "details": f"Missing {len(missing_files)} files. Found {len(found_files)} files ({total_size_gb:.1f}GB total)"
        }
    
    logger.info("All required model files found")
    return {
        "found": True, 
        "message": f"✅ Qwen-Image model found (snapshot {snapshot_id})", 
        "details": f"Complete model with {len(found_files)} files ({total_size_gb:.1f}GB total)"
    }

def load_existing_model():
    """Load model from existing cache"""
    global pipe, model_loaded
    
    logger.info("Attempting to load existing model...")
    device, torch_dtype = get_device_and_dtype()
    cache_dir = MODEL_DIR / "cache"
    
    try:
        logger.info("Loading pipeline components from cache...")
        
        # Comprehensive warning suppression for config warnings
        import transformers
        import diffusers.models
        import logging
        
        # Save original levels
        old_transformers_level = transformers.logging.get_verbosity()
        old_diffusers_level = logging.getLogger('diffusers').level
        
        # Suppress all warnings during model loading
        transformers.logging.set_verbosity_error()
        logging.getLogger('diffusers').setLevel(logging.ERROR)
        logging.getLogger('diffusers.models').setLevel(logging.ERROR)
        
        # Suppress Python warnings temporarily
        import warnings
        
        logger.info("Starting DiffusionPipeline.from_pretrained...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")

            logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
            logger.info("Loading may take several minutes for large model shards...")

            pipe = DiffusionPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                cache_dir=str(cache_dir),
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                resume_download=True  # Resume incomplete downloads
            )

            try:
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing")
            except Exception as e:
                logger.info(f"Attention slicing not available: {e}")

            if device == "cuda":
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                offload_dir = MODEL_DIR / "offload"
                offload_dir.mkdir(exist_ok=True)
                try:
                    if vram_gb >= MODEL_SIZE_GB:
                        pipe.to("cuda")
                        logger.info("Loaded model fully into GPU memory")
                        mode_msg = "GPU"
                    else:
                        pipe.enable_model_cpu_offload()
                        logger.info("Enabled model CPU offload due to limited VRAM")
                        mode_msg = "GPU with CPU offload"
                except RuntimeError as e:
                    logger.warning(f"GPU load failed ({e}), falling back to CPU offload")
                    pipe.enable_model_cpu_offload()
                    mode_msg = "GPU with CPU offload"
            else:
                pipe.to("cpu")
                mode_msg = "CPU"

        logger.info("DiffusionPipeline loaded successfully")

        # Restore logging levels
        transformers.logging.set_verbosity(old_transformers_level)
        logging.getLogger('diffusers').setLevel(old_diffusers_level)

        model_loaded = True
        logger.info(f"Model loaded successfully on {mode_msg}")
        return f"✅ Model loaded from cache using {mode_msg}"
        
    except Exception as e:
        error_msg = f"Failed to load existing model: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Provide more specific error information
        if "CUDA out of memory" in str(e):
            return f"❌ CUDA out of memory error. Try restarting the application or reducing system GPU usage."
        elif "No module named" in str(e):
            return f"❌ Missing dependencies: {str(e)}"
        elif "local_files_only" in str(e):
            return f"❌ Model files corrupted or incomplete. Please re-download the model."
        elif "safetensors" in str(e).lower() or "shard" in str(e).lower():
            return f"❌ Model file corruption detected. Please click 'Download & Load Model' to re-download."
        elif "memory" in str(e).lower() or "ram" in str(e).lower():
            return f"❌ Insufficient RAM. Need ~{MODEL_SIZE_GB}GB available RAM for model loading. Close other applications and try again."
        elif "killed" in str(e).lower() or "terminated" in str(e).lower():
            return f"❌ Process killed during loading - likely out of memory. Restart application and close other programs."
        else:
            return f"❌ Model loading error: {str(e)[:200]}..."

def download_model(model_variant, progress=gr.Progress()):
    """Download and initialize the Qwen-Image model"""
    global pipe, model_loaded

    set_model_variant(model_variant)
    logger.info(f"Starting model download/load process for {MODEL_NAME}...")
    
    if progress:
        progress(0, desc="Initializing...")
    
    # Check if model already exists and is complete
    model_status = check_model_files()
    if model_status["found"]:
        logger.info("Model files found, attempting to load from cache...")
        if progress:
            progress(0.2, desc="Loading existing model...")
        
        result = load_existing_model()
        if result and "✅" in result:  # Success message contains checkmark
            if progress:
                progress(1.0, desc="Model ready!")
            return result
        else:
            logger.warning("Failed to load from cache, will clear cache and re-download")
            # Clear potentially corrupted cache
            try:
                cache_dir = MODEL_DIR / "cache" / f"models--{MODEL_NAME.replace('/', '--')}"
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    logger.info("Cleared corrupted model cache")
            except Exception as e:
                logger.warning(f"Could not clear cache: {e}")
            
            if progress:
                progress(0.1, desc="Cache cleared, re-downloading...")
    
    logger.info("Downloading model files...")
    device, torch_dtype = get_device_and_dtype()
    
    try:
        # Download to custom cache directory
        cache_dir = MODEL_DIR / "cache"
        logger.info(f"Cache directory: {cache_dir}")

        if progress:
            progress(0.1, desc=f"Downloading {MODEL_NAME} model...")
        
        logger.info(f"Downloading {MODEL_NAME} with torch_dtype={torch_dtype}")
        
        # Comprehensive warning suppression for config warnings
        import transformers
        import diffusers.models
        import logging
        
        # Save original levels
        old_transformers_level = transformers.logging.get_verbosity()
        old_diffusers_level = logging.getLogger('diffusers').level
        
        # Suppress all warnings during model loading
        transformers.logging.set_verbosity_error()
        logging.getLogger('diffusers').setLevel(logging.ERROR)
        logging.getLogger('diffusers.models').setLevel(logging.ERROR)
        
        # Suppress Python warnings temporarily
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
            
            if progress:
                progress(0.3, desc="Downloading model files...")

            logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
            logger.info("Download may take 10-30 minutes depending on connection...")

            pipe = DiffusionPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                cache_dir=str(cache_dir),
                low_cpu_mem_usage=True,
                use_safetensors=True,
                resume_download=True,  # Resume interrupted downloads
                force_download=False   # Use cached files if available and valid
            )

            if progress:
                progress(0.7, desc="Setting up device...")

            try:
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing")
            except Exception as e:
                logger.info(f"Attention slicing not available: {e}")

            if device == "cuda":
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                offload_dir = MODEL_DIR / "offload"
                offload_dir.mkdir(exist_ok=True)
                try:
                    if vram_gb >= MODEL_SIZE_GB:
                        pipe.to("cuda")
                        logger.info("Loaded model fully into GPU memory")
                        mode_msg = "GPU"
                    else:
                        pipe.enable_model_cpu_offload()
                        logger.info("Enabled model CPU offload due to limited VRAM")
                        mode_msg = "GPU with CPU offload"
                except RuntimeError as e:
                    logger.warning(f"GPU load failed ({e}), falling back to CPU offload")
                    pipe.enable_model_cpu_offload()
                    mode_msg = "GPU with CPU offload"
            else:
                pipe.to("cpu")
                mode_msg = "CPU"
        
        # Restore logging levels
        transformers.logging.set_verbosity(old_transformers_level)
        logging.getLogger('diffusers').setLevel(old_diffusers_level)
        
        logger.info("Model files downloaded and loaded successfully")
        model_loaded = True

        if progress:
            progress(1.0, desc="Model ready!")

        result = f"✅ Model downloaded and loaded using {mode_msg}"
        logger.info(result)
        return result
        
    except Exception as e:
        error_msg = f"❌ Download/loading error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        
        # Provide more specific error information
        if "CUDA out of memory" in str(e):
            return f"❌ CUDA out of memory during download. Try restarting the application."
        elif "Connection" in str(e) or "timeout" in str(e).lower():
            return f"❌ Network connection error. Check internet connection and try again."
        elif "disk" in str(e).lower() or "space" in str(e).lower():
            return f"❌ Insufficient disk space. Need ~4GB free space for model download."
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            return f"❌ File permission error. Try running as administrator or check folder permissions."
        else:
            return f"❌ Download error: {str(e)[:200]}..."

def generate_image(prompt, negative_prompt, aspect_ratio, resolution, steps, cfg_scale, seed, randomize_seed, progress=gr.Progress()):
    """Generate image using Qwen-Image model"""
    global pipe, model_loaded
    
    if not model_loaded or pipe is None:
        error_msg = "❌ Model not loaded. Please download model first."
        logger.error(error_msg)
        return None, error_msg
    
    if not prompt.strip():
        error_msg = "❌ Please enter a prompt"
        logger.warning(error_msg)
        return None, error_msg
    
    logger.info(f"Starting image generation with prompt: '{prompt[:100]}...'")
    
    try:
        if progress:
            progress(0, desc="Preparing generation...")
        
        # Handle seed
        original_seed = seed
        if randomize_seed:
            seed = torch.randint(0, 1000000, (1,)).item()
            logger.info(f"Using randomized seed: {seed}")
        else:
            logger.info(f"Using fixed seed: {seed}")
        
        # Resolution mappings organized by aspect ratio
        resolution_data = {
            "1:1 (Square)": {
                "Small (512×512)": (512, 512),
                "Medium (1024×1024)": (1024, 1024),
                "Large (1328×1328)": (1328, 1328),
                "Max (2048×2048)": (2048, 2048),
            },
            "16:9 (Landscape)": {
                "Small (512×288)": (512, 288),
                "Medium (1024×576)": (1024, 576),
                "Large (1664×928)": (1664, 928),
                "Max (2048×1152)": (2048, 1152),
            },
            "9:16 (Portrait)": {
                "Small (288×512)": (288, 512),
                "Medium (576×1024)": (576, 1024),
                "Large (928×1664)": (928, 1664),
                "Max (1152×2048)": (1152, 2048),
            },
            "4:3 (Classic)": {
                "Small (512×384)": (512, 384),
                "Medium (1024×768)": (1024, 768),
                "Large (1472×1140)": (1472, 1140),
                "Max (2048×1536)": (2048, 1536),
            },
            "3:4 (Portrait)": {
                "Small (384×512)": (384, 512),
                "Medium (768×1024)": (768, 1024),
                "Large (1140×1472)": (1140, 1472),
                "Max (1536×2048)": (1536, 2048),
            },
            "3:2 (Photo)": {
                "Small (512×342)": (512, 342),
                "Medium (1024×682)": (1024, 682),
                "Large (1584×1056)": (1584, 1056),
                "Max (2048×1366)": (2048, 1366),
            },
            "2:3 (Portrait)": {
                "Small (342×512)": (342, 512),
                "Medium (682×1024)": (682, 1024),
                "Large (1056×1584)": (1056, 1584),
                "Max (1366×2048)": (1366, 2048),
            }
        }
        
        width, height = resolution_data[aspect_ratio][resolution]
        logger.info(f"Using resolution: {width}x{height} ({aspect_ratio})")
        
        # Add positive magic from official sample for better results
        positive_magic = "Ultra HD, 4K, cinematic composition."
        enhanced_prompt = f"{prompt}, {positive_magic}"
        logger.info(f"Enhanced prompt: '{enhanced_prompt[:150]}...'")
        
        neg_prompt = negative_prompt if negative_prompt.strip() else " "
        logger.info(f"Negative prompt: '{neg_prompt[:100]}...'")
        logger.info(f"Steps: {steps}, CFG Scale: {cfg_scale}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if progress:
            progress(0.2, desc=f"Generating image ({device.upper()})...")

        logger.info(f"Starting diffusion process on {device.upper()}...")
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=neg_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg_scale,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images[0]
        
        logger.info("Image generation completed")
        
        if progress:
            progress(0.9, desc="Saving image...")
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"qwen_image_{timestamp}_seed{seed}.png"
        image.save(filename)
        
        logger.info(f"Image saved to: {filename}")
        
        if progress:
            progress(1.0, desc="Complete!")
        
        result_info = f"✅ Image generated successfully!\nSeed: {seed}\nResolution: {width}x{height}\nSaved: {filename}"
        logger.info("Generation process completed successfully")
        
        return image, result_info
        
    except Exception as e:
        error_msg = f"❌ Generation error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        return None, error_msg

def clear_cache_and_redownload(model_variant):
    """Clear model cache for the selected variant"""
    global pipe, model_loaded

    set_model_variant(model_variant)
    logger.info("Manual cache clear requested")
    pipe = None
    model_loaded = False

    try:
        cache_dir = MODEL_DIR / "cache" / f"models--{MODEL_NAME.replace('/', '--')}"
        if cache_dir.exists():
            import shutil
            logger.info(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            logger.info("Model cache cleared successfully")
            return "🗑️ Cache cleared. Click 'Download & Load Model' to download fresh copy."
        else:
            return "ℹ️ No cache found to clear."
    except Exception as e:
        error_msg = f"❌ Failed to clear cache: {str(e)}"
        logger.error(error_msg)
        return error_msg


def check_startup_status(model_variant):
    """Check model status on startup for the selected variant"""
    set_model_variant(model_variant)
    model_status = check_model_files()

    if model_status["found"]:
        logger.info("Model files detected on startup")
        result = load_existing_model()
        if result and "✅" in result:  # Success message contains checkmark
            return result
        else:
            # Model found but failed to load
            return f"⚠️ {model_status['message']}\n{model_status['details']}\nClick 'Download & Load Model' to reload or re-download"
    else:
        # No model found or incomplete
        details = f" - {model_status['details']}" if model_status['details'] else ""
        return f"📥 {model_status['message']}{details}\nClick 'Download & Load Model' to download {MODEL_NAME} (~{MODEL_SIZE_GB}GB)"

# Create Gradio interface
with gr.Blocks(title="Qwen-Image GUI", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🎨 Qwen-Image Generator")
    gr.Markdown("Generate high-quality images using Alibaba's Qwen-Image diffusion model")
    gr.Markdown("💡 **Logs are saved to `console.log` - check this file for detailed progress info**")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Model Management
            with gr.Group():
                gr.Markdown("### 📦 Model Management")
                model_variant = gr.Dropdown(
                    label="Model Variant",
                    choices=list(MODEL_VARIANTS.keys()),
                    value=selected_variant,
                )

                download_btn = gr.Button("Download & Load Model", variant="primary", size="lg")

                with gr.Row():
                    clear_cache_btn = gr.Button("🗑️ Clear Cache & Re-download", variant="secondary", size="sm")

                model_status = gr.Textbox(
                    label="Status",
                    value=check_startup_status(selected_variant),
                    interactive=False,
                    lines=3
                )
            
            # Generation Controls
            with gr.Group():
                gr.Markdown("### 🎛️ Generation Settings")
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want in the image...",
                    lines=2
                )
                
                aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=[
                        "1:1 (Square)",
                        "16:9 (Landscape)", 
                        "9:16 (Portrait)",
                        "4:3 (Classic)",
                        "3:4 (Portrait)",
                        "3:2 (Photo)",
                        "2:3 (Portrait)"
                    ],
                    value="1:1 (Square)"
                )
                
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=[
                        "Small (512×512)",
                        "Medium (1024×1024)",
                        "Large (1328×1328)",
                        "Max (2048×2048)"
                    ],
                    value="Small (512×512)"
                )
                
                # Resolution display field
                resolution_display = gr.Textbox(
                    label="Output Resolution",
                    value="512 × 512 pixels",
                    interactive=False,
                    max_lines=1
                )
                
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50
                    )
                    
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        value=4.0
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0
                    )
                    
                    randomize_seed = gr.Checkbox(
                        label="Randomize",
                        value=True
                    )
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output
            with gr.Group():
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Result", show_download_button=True)
                generation_info = gr.Textbox(label="Generation Info", lines=4, interactive=False)
    
    def update_resolution_choices(aspect_ratio_choice):
        """Update resolution choices based on selected aspect ratio"""
        resolution_data = {
            "1:1 (Square)": [
                "Small (512×512)",
                "Medium (1024×1024)",
                "Large (1328×1328)",
                "Max (2048×2048)"
            ],
            "16:9 (Landscape)": [
                "Small (512×288)",
                "Medium (1024×576)",
                "Large (1664×928)",
                "Max (2048×1152)"
            ],
            "9:16 (Portrait)": [
                "Small (288×512)",
                "Medium (576×1024)",
                "Large (928×1664)",
                "Max (1152×2048)"
            ],
            "4:3 (Classic)": [
                "Small (512×384)",
                "Medium (1024×768)",
                "Large (1472×1140)",
                "Max (2048×1536)"
            ],
            "3:4 (Portrait)": [
                "Small (384×512)",
                "Medium (768×1024)",
                "Large (1140×1472)",
                "Max (1536×2048)"
            ],
            "3:2 (Photo)": [
                "Small (512×342)",
                "Medium (1024×682)",
                "Large (1584×1056)",
                "Max (2048×1366)"
            ],
            "2:3 (Portrait)": [
                "Small (342×512)",
                "Medium (682×1024)",
                "Large (1056×1584)",
                "Max (1366×2048)"
            ]
        }
        
        choices = resolution_data.get(aspect_ratio_choice, ["Small (512×512)"])
        default_value = choices[0]  # First option as default
        return gr.update(choices=choices, value=default_value)
    
    def update_resolution_display(resolution_choice):
        """Extract and display resolution from resolution choice"""
        # Extract resolution from choice like "Small (512×512)"
        if "(" in resolution_choice and "×" in resolution_choice:
            resolution_part = resolution_choice.split("(")[1].split(")")[0]
            width, height = resolution_part.split("×")
            return f"{width} × {height} pixels"
        return "Resolution not available"
    
    # Event handlers
    download_btn.click(
        fn=download_model,
        inputs=[model_variant],
        outputs=[model_status],
        show_progress=True
    )

    clear_cache_btn.click(
        fn=clear_cache_and_redownload,
        inputs=[model_variant],
        outputs=[model_status],
        show_progress=False
    )

    model_variant.change(
        fn=check_startup_status,
        inputs=[model_variant],
        outputs=[model_status]
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, aspect_ratio, resolution, steps, cfg_scale, seed, randomize_seed],
        outputs=[output_image, generation_info],
        show_progress=True
    )
    
    # Update resolution choices when aspect ratio changes
    aspect_ratio.change(
        fn=update_resolution_choices,
        inputs=[aspect_ratio],
        outputs=[resolution]
    )
    
    # Update resolution display when resolution changes
    resolution.change(
        fn=update_resolution_display,
        inputs=[resolution],
        outputs=[resolution_display]
    )

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Starting Qwen-Image GUI")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"Model cache directory: {MODEL_DIR.absolute()}")
    logger.info(f"Log file: {LOG_FILE.absolute()}")
    logger.info("="*50)
    
    try:
        # Completely disable problematic loggers
        import logging
        
        # More aggressive logger disabling
        loggers_to_disable = [
            'uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.config',
            'gradio', 'gradio.routes', 'gradio.utils', 'fastapi', 'starlette'
        ]
        
        for logger_name in loggers_to_disable:
            logging.getLogger(logger_name).disabled = True
            logging.getLogger(logger_name).handlers = []
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        logger.info("Launching Gradio interface...")
        
        # Use the most basic launch configuration
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            quiet=True
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
        
    except Exception as e:
        error_msg = f"Application error: {e}"
        logger.error(error_msg)
        
        # For any error, try a fallback launch method with minimal parameters
        logger.info("Attempting fallback Gradio launch...")
        try:
            # Most basic launch possible
            interface.launch(
                server_port=7861,  # Try different port
                share=False
            )
        except Exception as fallback_error:
            logger.error(f"Fallback launch also failed: {fallback_error}")
            
            # Last resort: try with absolutely minimal settings
            logger.info("Attempting final basic launch...")
            try:
                interface.launch()
            except Exception as final_error:
                logger.error(f"Final launch attempt failed: {final_error}")
                logger.info("Model is loaded and ready - you may need to restart to access the GUI")
            
    finally:
        # Log final message before cleanup
        logger.info("Application shutting down")
        # Cleanup will be handled by atexit handler