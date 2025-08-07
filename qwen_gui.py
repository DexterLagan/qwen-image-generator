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
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Suppress specific Gradio warnings early
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Set CUDA memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configure paths
MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
LOG_FILE = Path("console.log")
MODEL_NAME = "Qwen/Qwen-Image"

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
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"CUDA device detected: {device_name}")
        logger.info(f"GPU VRAM: {vram_gb:.1f}GB available")
        
        # Check if we have enough VRAM for the full model (need ~20-25GB)
        if vram_gb < 20:
            logger.warning(f"VRAM ({vram_gb:.1f}GB) insufficient for full model. Will use CPU offloading with GPU compute.")
            return "cuda", torch.bfloat16  # Still use CUDA for compute, but offload to CPU memory
        else:
            return "cuda", torch.bfloat16
    else:
        logger.info("No CUDA device detected, using CPU")
        if memory.available < 8 * (1024**3):  # Less than 8GB available
            logger.warning(f"Low available RAM ({memory.available / (1024**3):.1f}GB). Model loading may be slow or fail.")
        return "cpu", torch.float32

def check_model_files():
    """Check if model files are already downloaded and complete"""
    cache_dir = MODEL_DIR / "cache" / "models--Qwen--Qwen-Image"
    logger.info(f"Checking for model files in: {cache_dir}")
    
    if not cache_dir.exists():
        logger.info("Model cache directory not found")
        return False
    
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
        return False
    
    # Find the latest snapshot
    snapshots = list(snapshot_dir.iterdir())
    if not snapshots:
        logger.info("No snapshots found")
        return False
    
    latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
    logger.info(f"Checking snapshot: {latest_snapshot.name}")
    
    missing_files = []
    for file_path in required_files:
        full_path = latest_snapshot / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    # Check for split model files using glob patterns
    for pattern in required_patterns:
        pattern_path = str(latest_snapshot / pattern)
        matching_files = glob.glob(pattern_path)
        if not matching_files:
            missing_files.append(pattern)
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
        return False
    
    logger.info("All required model files found")
    return True

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
        
        # Suppress Python warnings temporarily (but don't redirect stderr as it breaks model loading)
        import warnings
        
        logger.info("Starting DiffusionPipeline.from_pretrained with CPU offloading...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
            
            # Use CPU offloading for large models to prevent RAM issues
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                cache_dir=str(cache_dir),
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Enable memory optimization - use CPU offloading for large models
            logger.info("Enabling CPU offloading to manage large model...")
            pipe.enable_sequential_cpu_offload()
            
            # Also enable memory efficient attention if available
            try:
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing for additional memory savings")
            except Exception as e:
                logger.info(f"Attention slicing not available: {e}")
            
            try:
                # Enable CUDA memory optimization
                if device == "cuda":
                    torch.cuda.empty_cache()  # Clear any existing GPU memory
                    logger.info("Cleared GPU memory cache")
            except Exception as e:
                logger.info(f"GPU memory optimization failed: {e}")
        
        logger.info("DiffusionPipeline loaded successfully")
        
        # Restore logging levels
        transformers.logging.set_verbosity(old_transformers_level)
        logging.getLogger('diffusers').setLevel(old_diffusers_level)
        
        # Skip explicit .to(device) when using CPU offloading as it's handled automatically
        # CPU offloading manages device placement automatically
        logger.info("Using CPU offloading - device placement handled automatically")
        model_loaded = True
        logger.info(f"Model loaded successfully on {device.upper()}")
        return f"✅ Model loaded from cache on {device.upper()}"
    except Exception as e:
        logger.error(f"Failed to load existing model: {str(e)}")
        return None

def download_model(progress=gr.Progress()):
    """Download and initialize the Qwen-Image model"""
    global pipe, model_loaded
    
    logger.info("Starting model download/load process...")
    
    if progress:
        progress(0, desc="Initializing...")
    
    # Check if model already exists and is complete
    if check_model_files():
        logger.info("Model files found, attempting to load from cache...")
        if progress:
            progress(0.2, desc="Loading existing model...")
        
        result = load_existing_model()
        if result:
            if progress:
                progress(1.0, desc="Model ready!")
            return result
        else:
            logger.warning("Failed to load from cache, will re-download")
    
    logger.info("Downloading model files...")
    device, torch_dtype = get_device_and_dtype()
    
    try:
        # Download to custom cache directory
        cache_dir = MODEL_DIR / "cache"
        logger.info(f"Cache directory: {cache_dir}")
        
        if progress:
            progress(0.1, desc="Starting download...")
        
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
        
        # Suppress Python warnings temporarily (but don't redirect stderr as it breaks model loading)
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
            
            # Download model files with CPU offloading
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch_dtype,
                cache_dir=str(cache_dir),
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Enable memory optimization - use CPU offloading for large models
            logger.info("Enabling CPU offloading to manage large model...")
            pipe.enable_sequential_cpu_offload()
            
            # Also enable memory efficient attention if available
            try:
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing for additional memory savings")
            except Exception as e:
                logger.info(f"Attention slicing not available: {e}")
            
            try:
                # Enable CUDA memory optimization
                if device == "cuda":
                    torch.cuda.empty_cache()  # Clear any existing GPU memory
                    logger.info("Cleared GPU memory cache")
            except Exception as e:
                logger.info(f"GPU memory optimization failed: {e}")
        
        # Restore logging levels
        transformers.logging.set_verbosity(old_transformers_level)
        logging.getLogger('diffusers').setLevel(old_diffusers_level)
        
        logger.info("Model files downloaded and loaded successfully")
        
        if progress:
            progress(0.8, desc="Loading model to device...")
        
        # Skip explicit .to(device) when using CPU offloading as it's handled automatically
        # CPU offloading manages device placement automatically
        logger.info("Using CPU offloading - device placement handled automatically")
        model_loaded = True
        
        logger.info("Model loaded to device successfully")
        
        if progress:
            progress(1.0, desc="Model ready!")
        
        result = f"✅ Model downloaded and loaded on {device.upper()}"
        logger.info(result)
        return result
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        return error_msg

def generate_image(prompt, negative_prompt, aspect_ratio, steps, cfg_scale, seed, randomize_seed, progress=gr.Progress()):
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
        
        # Aspect ratio mappings from official sample
        aspect_ratios = {
            "1:1 (Square)": (1328, 1328),
            "16:9 (Landscape)": (1664, 928),
            "9:16 (Portrait)": (928, 1664),
            "4:3 (Classic)": (1472, 1140),
            "3:4 (Portrait)": (1140, 1472),
            "3:2 (Photo)": (1584, 1056),
            "2:3 (Portrait)": (1056, 1584),
        }
        
        width, height = aspect_ratios[aspect_ratio]
        logger.info(f"Using resolution: {width}x{height} ({aspect_ratio})")
        
        # Add positive magic from official sample for better results
        positive_magic = "Ultra HD, 4K, cinematic composition."
        enhanced_prompt = f"{prompt}, {positive_magic}"
        logger.info(f"Enhanced prompt: '{enhanced_prompt[:150]}...'")
        
        neg_prompt = negative_prompt if negative_prompt.strip() else " "
        logger.info(f"Negative prompt: '{neg_prompt[:100]}...'")
        logger.info(f"Steps: {steps}, CFG Scale: {cfg_scale}")
        
        device, _ = get_device_and_dtype()
        
        if progress:
            progress(0.2, desc="Generating image...")
        
        logger.info("Starting diffusion process...")
        # Generate image
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

def check_startup_status():
    """Check model status on startup"""
    if check_model_files():
        logger.info("Model files detected on startup")
        result = load_existing_model()
        if result:
            return result
    return "Model not loaded - click 'Download & Load Model' to begin"

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
                download_btn = gr.Button("Download & Load Model", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="Status", 
                    value=check_startup_status(), 
                    interactive=False,
                    lines=2
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
    
    # Event handlers
    download_btn.click(
        fn=download_model,
        outputs=[model_status],
        show_progress=True
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, aspect_ratio, steps, cfg_scale, seed, randomize_seed],
        outputs=[output_image, generation_info],
        show_progress=True
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