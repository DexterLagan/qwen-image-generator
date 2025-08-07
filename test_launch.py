#!/usr/bin/env python3

print("Starting Qwen-Image GUI test...")

try:
    import gradio as gr
    import torch
    from diffusers import DiffusionPipeline
    from pathlib import Path
    
    print("All imports successful")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Gradio version: {gr.__version__}")
    
    # Test basic Gradio interface
    def hello(name):
        return f"Hello {name}!"
    
    interface = gr.Interface(
        fn=hello,
        inputs=gr.Textbox(label="Name"),
        outputs=gr.Textbox(label="Greeting"),
        title="Qwen-Image GUI Test"
    )
    
    print("Gradio interface created successfully")
    print("Starting server on http://localhost:7860")
    
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")