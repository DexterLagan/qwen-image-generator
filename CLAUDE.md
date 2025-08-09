# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QwenImageTool is a **working prototype** for a cross-platform desktop application that provides a GUI for Alibaba's Qwen-Image diffusion model. This project exists because Qwen-Image is a new model that does not yet work with ComfyUI, so a standalone GUI solution is needed.

**Current Status**: Fully functional Gradio prototype with advanced memory management - ready for production use and future Tauri wrapper development.

The working architecture combines:
- **Python backend**: Gradio interface (`qwen_gui.py`) with intelligent memory management
- **Tauri frontend**: Rust-based desktop wrapper for native app distribution (planned)
- **Model**: Qwen/Qwen-Image from Hugging Face (https://huggingface.co/Qwen/Qwen-Image), with automatic download and caching
- **Advanced Features**: CPU offloading, GPU compute, smart device detection, comprehensive logging

## Development Commands

### Python Environment

**Windows Setup Notes:**
- **Recommended**: Use PowerShell instead of CMD for better Unicode support and Python integration
- **PATH Issues**: pip may install to user directory (`%APPDATA%\Roaming\Python\Python313\site-packages`) vs system directory
- **Unicode**: CMD has encoding issues with Unicode characters - PowerShell handles this better

```powershell
# Install Python dependencies (CRITICAL: uses diffusers, not transformers)
pip install -r requirements.txt

# Test basic functionality first
python test_launch.py

# Run the full Qwen-Image GUI
python qwen_gui.py
```

**Dependencies installed:**
- `torch>=2.0.0` - Deep learning framework
- `git+https://github.com/huggingface/diffusers` - Latest diffusers with Qwen-Image support  
- `gradio>=4.0.0` - Web UI framework
- `accelerate` - Model acceleration utilities
- `Pillow` - Image processing

### Tauri Commands (when implemented)
```bash
# Development mode with hot reload
npm run tauri dev

# Production build
npm run tauri build
```

## Architecture

### Current Implementation
- **Single file architecture**: All functionality is contained in `qwen_gui.py` (NEEDS REWRITE)
- **Model loading**: ❌ Currently uses `transformers.AutoModelForCausalLM` (INCORRECT)
- **Correct approach**: ✅ Should use `diffusers.DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")`
- **Image generation**: Handles text-to-image generation with configurable parameters
- **Output management**: Saves generated images to local `output/` directory with timestamps

### Key Components in qwen_gui.py (NEEDS COMPLETE REWRITE)
- ❌ Model initialization (lines 9-12): Currently uses wrong API (`AutoModelForCausalLM`)
- ✅ Resolution/aspect ratio mappings (lines 19-30): Correct concept, but resolutions are too low
- ❌ `generate_image()` function (lines 32-73): Uses wrong API calls
- ✅ Gradio interface (lines 76-95): UI structure is good, but parameters need adjustment

### Correct Implementation (see qwen_sample_hf.py)
- **Library**: `from diffusers import DiffusionPipeline`
- **Model loading**: `DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)`
- **Generation**: `pipe(prompt, negative_prompt, width, height, num_inference_steps, true_cfg_scale, generator)`
- **Parameters**: Uses `true_cfg_scale` instead of `guidance_scale`
- **Resolutions**: Higher resolutions (1328x1328 for 1:1, 1664x928 for 16:9)

### Model Management
- **Model variants**:
  - FP8 (~20GB): https://huggingface.co/Qwen/Qwen-Image
  - FP16 (~40GB): https://huggingface.co/Qwen/Qwen-Image-FP16
- **Automatic download**: Downloads selected variant on first run
- **Platform-specific caching**: Model files stored in appropriate user cache directories:
  - Windows: `%LOCALAPPDATA%\QwenImage\models`
  - macOS: `~/Library/Application Support/QwenImage/models`
  - Linux: `~/.local/share/QwenImage/models`
- **Hardware optimization**: Uses GPU (CUDA) if available, falls back to CPU
- **Persistent storage**: Model files cached by Transformers library for subsequent runs

### File Structure
```
QwenImageTool/
├── qwen_gui.py                           # ✅ NEW: Proper Gradio application with diffusers
├── qwen_sample_hf.py                     # 📖 Official HuggingFace sample code (reference)
├── qwen_gradio_proto_chatgpt_wrong.py    # ❌ OLD: Wrong ChatGPT prototype (transformers)
├── test_launch.py                        # 🧪 Basic Gradio test script
├── requirements.txt                      # 📦 Python dependencies
├── README.md                             # 📋 Project specification
├── model/cache/                          # 🤖 Downloaded model files (auto-created)
└── output/                               # 🖼️ Generated images (auto-created)
```

## Development Notes

- **Current status**: Proper diffusers-based prototype completed and ready for testing
- **Dependencies installed**: All required packages (torch, diffusers, gradio) are installed
- **Windows environment**: Tested on Windows with Python 3.13 - requires PowerShell for best results
- **Model compatibility**: Built specifically for Qwen-Image since it's not yet supported in ComfyUI
- **Model download**: Automatic download to `model/cache/` on first use (several GB download)
- **Local output**: Images saved to `output/` directory with timestamp and seed information
- **GPU support**: Automatically detects CUDA availability, falls back to CPU
- **UI features**: 
  - Separate model download step with progress tracking
  - Official aspect ratios and resolutions from HuggingFace sample
  - Real-time generation progress and status updates

### Current Status & Completed Features
1. ✅ **COMPLETED**: Proper `qwen_gui.py` using `diffusers.DiffusionPipeline` API 
2. ✅ **COMPLETED**: Uses `true_cfg_scale` and official HuggingFace parameters
3. ✅ **COMPLETED**: Official resolution presets (1328x1328, 1664x928, etc.)
4. ✅ **COMPLETED**: Advanced memory management with CPU offloading
5. ✅ **COMPLETED**: Smart GPU/CPU detection with automatic fallback
6. ✅ **COMPLETED**: CUDA-enabled PyTorch with proper driver support
7. ✅ **COMPLETED**: Comprehensive logging system with file output
8. ✅ **COMPLETED**: Gradio compatibility fixes and stable server launch
9. ✅ **COMPLETED**: Model file verification and smart caching
10. ✅ **COMPLETED**: Progressive error handling and fallback systems

### Ready for Production
- **Fully functional**: GUI works with model loading, generation, and file saving
- **Memory optimized**: Handles large models on systems with limited VRAM
- **Cross-platform**: Windows tested, Linux/macOS compatible
- **Future-ready**: Architecture ready for Tauri desktop wrapper

## Advanced Features & Technical Achievements

### Memory Management System
- **CPU Offloading**: Automatically stores large models in RAM while using GPU for computation
- **Smart Device Detection**: Detects VRAM capacity and chooses optimal loading strategy
- **Attention Slicing**: Reduces memory usage during image generation
- **Dynamic Memory Management**: Clears GPU cache and optimizes allocation patterns
- **Progressive Loading**: Loads model components sequentially to minimize peak memory usage

### Compatibility & Stability
- **PyTorch CUDA Integration**: Proper CUDA-enabled PyTorch installation with version compatibility
- **TorchVision Compatibility**: Fixed version conflicts between torch/torchvision/diffusers
- **Gradio Logging Fixes**: Resolved stdout redirection conflicts that broke internal formatters
- **Progressive Fallback**: Multiple launch attempts with increasingly minimal configurations
- **Cross-Platform Logging**: File-based logging that works on Windows/Linux/macOS

### Error Handling & Debugging
- **Comprehensive Logging**: Dual console/file output for debugging and monitoring
- **Exception Handling**: Global exception capture with detailed stack traces
- **Memory Monitoring**: Real-time RAM/VRAM usage reporting and warnings
- **Model Verification**: Smart detection of complete vs incomplete model downloads
- **Graceful Degradation**: Automatic fallback from GPU to CPU when needed

### Performance Optimizations
- **Model Caching**: Intelligent reuse of downloaded model files
- **Split File Support**: Handles large models split across multiple .safetensors files
- **Environment Variables**: CUDA memory optimization settings for better allocation
- **Lazy Loading**: Components loaded only when needed to reduce startup time

## Lessons Learned & Solutions

### Critical Issues Solved
1. **Model API Mismatch**: Original prototype used wrong `transformers` API instead of `diffusers`
2. **Memory Limitations**: 40GB+ model won't fit in 16GB VRAM - solved with CPU offloading
3. **PyTorch Installation**: Default pip install gives CPU-only version - need CUDA-specific install
4. **Version Conflicts**: TorchVision 0.2.0 missing modern features - fixed with compatible versions
5. **Gradio Logging**: stdout redirection broke internal logging - solved with proper logging setup
6. **Windows Environment**: CMD has Unicode/process issues - PowerShell recommended

### Key Technical Decisions
- **diffusers over transformers**: Correct API for Qwen-Image model
- **CPU offloading**: Enables GPU acceleration with RAM storage for large models
- **Sequential offloading**: Better than model CPU offloading for extreme memory constraints
- **File-based logging**: More reliable than stdout redirection for debugging
- **Smart device detection**: Automatic optimization based on available hardware

### Development Environment Requirements
- **PowerShell over CMD**: Better Unicode support and Python integration on Windows
- **CUDA PyTorch**: Essential for GPU acceleration, requires specific installation method
- **Accelerate library**: Enables low_cpu_mem_usage for efficient model loading
- **Latest diffusers**: Need development version from Git for Qwen-Image support

## Advanced Lessons Learned (Latest Session)

### **CRITICAL: Windows Page File Configuration**
**Issue**: Model loading crashes at "Loading checkpoint shards: 6/9" with process termination
**Root Cause**: Windows virtual memory insufficient for large model shard loading (even with 32GB RAM)
**Solution**: Increase Windows page file to 50GB via `sysdm.cpl` → Advanced → Performance → Virtual Memory
**Impact**: Without this fix, model loading will fail regardless of available physical RAM

### **Complete CPU-Only Implementation Required**
**Issue**: CUDA Out of Memory errors persist even with CPU offloading methods
**Root Cause**: `enable_sequential_cpu_offload()` and `enable_model_cpu_offload()` still attempt CUDA operations internally
**Solution**: Complete elimination of all GPU operations:
- Force `device="cpu"` and `torch_dtype=torch.float32`
- Never call `.to("cuda")` or any CPU offloading methods
- Use only CPU-safe optimizations like `enable_attention_slicing()`

### **Meta Tensor Error Resolution**
**Issue**: "Cannot copy out of meta tensor; no data!" NotImplementedError during generation
**Root Cause**: Mixed GPU/CPU operations create meta tensors without materialized data
**Solution**: Pure CPU-only pipeline with CPU-only generator: `torch.Generator(device="cpu")`

### **Safetensors Corruption Handling**
**Issue**: Model files become corrupted during interrupted downloads or loading failures
**Implementation**: 
- Auto-detect corruption patterns ("safetensors", "shard", "killed", "memory")
- Automatic cache clearing and re-download on detected corruption
- Manual "Clear Cache & Re-download" button for user recovery
- Resume interrupted downloads with `resume_download=True`

### **Memory Requirements & Monitoring**
**Findings**: 
- Model requires ~20GB RAM during loading (not just final model size)
- Windows needs additional virtual memory buffer during shard loading
- Real-time RAM monitoring essential for troubleshooting
- Process can be killed by system without error messages if insufficient virtual memory

### **Enhanced Error Categorization**
**Implementation**: Specific error detection and user-friendly messages:
- CUDA OOM errors → Suggest application restart
- Network/timeout errors → Check internet connection  
- Disk space errors → Request more storage space
- Permission errors → Suggest administrator mode
- Corruption errors → Trigger automatic cache clearing

## Latest UI and Platform Improvements

### **CUDA Platform Compatibility Fix**
**Issue**: `expandable_segments not supported on this platform` warning during model loading on Windows
**Root Cause**: PyTorch CUDA memory configuration not supported on all platforms
**Solution**: Removed `PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'` environment variable
**Impact**: Clean console output without platform-specific warnings

### **Improved Resolution UI System**
**Issue**: Combined "Resolution & Aspect Ratio" dropdown was confusing and didn't clearly show resolution options
**Implementation**: Split into intuitive two-dropdown system:

**New UI Structure**:
1. **Aspect Ratio Dropdown**: Clean list of ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
2. **Resolution Dropdown**: Dynamic options based on selected aspect ratio (Small, Medium, Large, Max)
3. **Resolution Display**: Read-only field showing actual pixels (e.g., "512 × 512 pixels")

**Dynamic Functionality**:
- Aspect ratio selection automatically updates available resolutions
- Resolution selection updates the pixel display in real-time
- All 28 resolution combinations preserved from previous system

**Resolution Matrix Maintained**:
- **1:1 Square**: 512², 1024², 1328² (Qwen default), 2048²
- **16:9 Landscape**: 512×288, 1024×576, 1664×928, 2048×1152
- **9:16 Portrait**: 288×512, 576×1024, 928×1664, 1152×2048
- **4:3 Classic**: 512×384, 1024×768, 1472×1140, 2048×1536
- **3:4 Portrait**: 384×512, 768×1024, 1140×1472, 1536×2048
- **3:2 Photo**: 512×342, 1024×682, 1584×1056, 2048×1366
- **2:3 Portrait**: 342×512, 682×1024, 1056×1584, 1366×2048

**User Experience Benefits**:
- Clear separation of concerns (aspect ratio vs resolution)
- Immediate visual feedback of output dimensions
- Intuitive workflow: choose shape first, then quality level
- Prevents confusion between aspect ratio and pixel dimensions

### Tauri Integration Plan
For the standalone desktop version:
1. **Tauri setup**: Initialize Tauri project with embedded Python runtime
2. **Backend launcher**: Create platform-specific scripts (run_backend.sh/bat) to start Python server
3. **WebView integration**: Configure Tauri to load Gradio interface from localhost:7860
4. **Model download UX**: Implement loading screen during first-run model download
5. **Bundling**: Include Python dependencies and ensure cross-platform compatibility