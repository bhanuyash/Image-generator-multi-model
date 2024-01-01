# Image Generation Tool with Multi-Model Support

### Introduction
This image generation tool is a personal project that showcases my coding abilities. It now supports multiple state-of-the-art models, including Stable Diffusion XL (SDXL), DALL-E, and Dreamshaper XL. Utilizing resources from Hugging Face and other platforms, this tool is ideal for anyone interested in AI-powered image generation. Note: Model downloads and proper CUDA setup are necessary for GPU usage.

### Model Information
- **Developers**: Stability AI, OpenAI, and others
- **Model Types**: Diverse, including Latent and DALL-E Diffusion Models
- **License**: Varies by model, typically open source
- **Description**: Capable of generating high-quality images from text prompts using various models.
- **Resources**: 
  - [Stability AI GitHub](https://github.com/Stability-AI/generative-models)
  - [SDXL Report on arXiv](https://arxiv.org/abs/2307.01952)
  - [Hugging Face SDXL repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - [Hugging Face OpenDalle repo](https://huggingface.co/dataautogpt3/OpenDalleV1.1)
  - [Hugging Face Dreamshaper XL repo](https://huggingface.co/Lykon/dreamshaper-xl-turbo)

![SDXL Model Visualization](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/pipeline.png)

## Updates
- **2024-01-01**: Enhanced the tool to allow the selection of multiple models and samplers. Added functionality to use custom, checkpoint downloaded SDXL models.

## Instructions for Use
### System Requirements
- **CUDA Toolkit**: Ensure you have the correct version installed for optimal performance.
- **cuDNN**: Required for deep neural network computations.
- **PyTorch with CUDA**: Match your CUDA version for GPU acceleration.

### Installation and Setup
1. **Install Required Libraries**: Upgrade `diffusers` to version 0.19.0 or higher and install additional dependencies.
   ```shell
   pip install diffusers --upgrade
   pip install invisible_watermark transformers accelerate safetensors

#### Running on CPU
- The code can also be run on a CPU if GPU resources are limited.
- Modify the code to offload model computations to the CPU by replacing `pipe.to("cuda")` with `pipe.enable_model_cpu_offload()`.

```python
# Replace this line for CPU usage
pipe.enable_model_cpu_offload()
```

## Demonstration
A demo of how the generator works

![Demo](https://i.imgur.com/89Yapc9.gif)

## Future Work
I inted to make additions to this project, one thing I want to include is user training (mainly fine-tuning, few shot or seed image). Will see. 
