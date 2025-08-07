# Qwen-Image Generator

A cross-platform desktop application providing a user-friendly GUI for Alibaba's Qwen-Image diffusion model. Built with Python/Gradio backend and designed for future Tauri desktop distribution.

![Qwen-Image Generator Screenshot](screenshots/screenshot.png)

## 🚀 Features

- **🎨 Easy Image Generation**: Simple web interface for text-to-image generation
- **🧠 Smart Memory Management**: CPU offloading for large models with GPU acceleration
- **⚡ Auto-Configuration**: Automatic GPU/CPU detection and optimization
- **📦 One-Click Setup**: Automatic model download and caching
- **🔧 Advanced Controls**: Multiple aspect ratios, CFG scale, steps, and seed controls
- **📊 Real-Time Monitoring**: Comprehensive logging and progress tracking
- **🌐 Cross-Platform**: Works on Windows, macOS, and Linux

## 🖼️ Generated Image Quality

The Qwen-Image model produces high-quality images with:
- **High Resolutions**: Up to 1664x928 (16:9) and 1328x1328 (1:1)
- **Multiple Aspect Ratios**: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
- **Advanced Controls**: CFG scale, negative prompts, custom seeds
- **Professional Quality**: 4K, ultra HD, cinematic composition enhancement

## 🛠️ Quick Start

### Prerequisites
- **Python 3.10+** with pip
- **PowerShell** (Windows) for best compatibility
- **16GB+ RAM** recommended
- **NVIDIA GPU with 8GB+ VRAM** (optional, but recommended)

### Installation

1. **Clone the repository**:
```powershell
git clone https://github.com/DexterLagan/qwen-image-generator.git
cd qwen-image-generator
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

3. **Launch the application**:
```powershell
python qwen_gui.py
```

4. **Open your browser** to `http://localhost:7860`

5. **First run**: Click "Download & Load Model" (one-time ~4GB download)

## 🧠 Advanced Memory Management

### Automatic Optimization
The application automatically detects your hardware and optimizes accordingly:

- **GPU Available (16GB+ VRAM)**: Full GPU acceleration
- **GPU Available (<16GB VRAM)**: CPU offloading with GPU compute
- **CPU Only**: Sequential CPU offloading with memory optimization

### Features
- **CPU Offloading**: Stores large models in RAM, uses GPU for computation
- **Attention Slicing**: Reduces memory usage during generation  
- **Smart Caching**: Reuses downloaded model files
- **Memory Monitoring**: Real-time RAM/VRAM usage reporting

## 📁 Project Structure

```
qwen-image-generator/
├── qwen_gui.py                           # 🚀 Main Gradio application
├── qwen_sample_hf.py                     # 📖 Official HuggingFace reference
├── qwen_gradio_proto_chatgpt_wrong.py    # 📁 Legacy prototype (archived)
├── test_launch.py                        # 🧪 Basic functionality test
├── requirements.txt                      # 📦 Python dependencies
├── console.log                           # 📋 Application logs
├── model/cache/                          # 🤖 Downloaded model files
├── output/                               # 🖼️ Generated images
└── screenshots/                          # 📸 Documentation images
```

## ⚙️ Technical Details

### Model Integration
- **API**: Uses `diffusers.DiffusionPipeline` (not transformers)
- **Model**: Qwen/Qwen-Image from Hugging Face
- **Parameters**: Official `true_cfg_scale`, aspect ratios, and resolutions
- **Caching**: Intelligent model file verification and reuse

### Compatibility
- **PyTorch**: CUDA-enabled version for GPU acceleration
- **Gradio**: Latest version with compatibility fixes
- **Cross-Platform**: Windows tested, Linux/macOS compatible

### Performance Optimizations
- **Environment Variables**: CUDA memory optimization settings
- **Progressive Loading**: Components loaded sequentially
- **Error Recovery**: Multiple fallback strategies for reliability

## 🔧 Configuration

### Environment Variables
The application sets optimal CUDA configurations automatically:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### Model Storage
- **Windows**: `model/cache/models--Qwen--Qwen-Image/`
- **Output**: `output/qwen_image_YYYYMMDD_HHMMSS_seedXXXXXX.png`

## 🚀 Future Development

### Planned Tauri Integration
- **Native Desktop App**: Rust-based wrapper for system integration
- **Embedded Runtime**: Self-contained Python environment
- **Cross-Platform Distribution**: .exe, .app, and .deb packages
- **Enhanced UX**: Native file dialogs, system tray, auto-updater

### Roadmap
- [ ] Tauri desktop wrapper implementation
- [ ] Batch image generation
- [ ] Image gallery and history
- [ ] Advanced prompt templates
- [ ] Model fine-tuning support

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Ensure you have the latest diffusers from requirements.txt
- **Memory Errors**: The app automatically enables CPU offloading for large models
- **GPU Not Detected**: Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Debug Information
- Check `console.log` for detailed operation logs
- Use PowerShell on Windows for better Unicode support
- Ensure 16GB+ RAM for optimal performance

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

### Model License
The Qwen-Image model is licensed under the Qianwen License 1.0:
- ✅ **Free** for personal, research, and evaluation use
- ❌ **Commercial use** requires separate license from Alibaba

## 👏 Credits

- **Model**: Alibaba Qwen Team ([Qwen-Image](https://huggingface.co/Qwen/Qwen-Image))
- **Framework**: Hugging Face ([Diffusers](https://github.com/huggingface/diffusers), [Gradio](https://gradio.app/))
- **Future Desktop**: [Tauri](https://tauri.app/)
- **Author**: Dexter Santucci

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

If you encounter any issues:
1. Check the `console.log` file for error details
2. Ensure you're using PowerShell on Windows
3. Verify your Python and dependency versions
4. Open an issue on GitHub with your log file