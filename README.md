# Image Generation Tool Using Stable Diffusion XL

### Introduction
I built this image generation tool as a hobby project to showcase my coding skills. It utilizes the Stable Diffusion XL model, a state-of-the-art, diffusion-based text-to-image generative model developed by Stability AI. I mainly utilised resources from hugging face to implement the model, anyone interested can fork the repo and run as is for a simple image generator. Also note that the models obviously will have to be downloaded and CUDA setup properly if you wan tto use your GPU.

### Model Information
- **Developer**: Stability AI
- **Model Type**: Latent Diffusion Model
- **License**: CreativeML Open RAIL++-M License
- **Description**: Capable of generating and modifying images from text prompts.
- **Resources**: [GitHub Repository](https://github.com/Stability-AI/generative-models), [SDXL Report on arXiv](https://arxiv.org/abs/2307.01952), [Hugging Face repo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

![Model Visualization](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/pipeline.png)

## Instructions for Use
#### System Requirements
- **CUDA Toolkit**: Ensure the right version of CUDA is installed for optimal performance. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- **cuDNN**: For deep neural network operations. [cuDNN](https://developer.nvidia.com/cudnn)
- **PyTorch with CUDA**: Specific to your CUDA version. [PyTorch with CUDA](https://pytorch.org/get-started/locally/)

#### Installation and Setup
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
