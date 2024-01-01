from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image,  DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import os
import sys
import threading
from ttkbootstrap import Style

class ConsoleRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, output):
        self.text_widget.insert(tk.END, output)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

def save_image(image, path='Image-generator-stable-diffusion/Images'):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(os.path.join(path, "generated_image.png"))

def initialize_model(model_name, custom_path=None):
    global pipe
    if model_name == "Stable Diffusion":
        #Intitialise StabilityAI's Stable Diffusion XL pipeline
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        
    elif model_name == "DALL-E":
        # Initialize DALL-E Pipeline
        pipe = AutoPipelineForText2Image.from_pretrained("dataautogpt3/OpenDalleV1.1", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        
    elif model_name == "DreamShaper XL":
        pipe = AutoPipelineForText2Image.from_pretrained('lykon/dreamshaper-xl-turbo', torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        
    elif model_name == "Custom Path":  
        
        pipe = StableDiffusionXLPipeline.from_single_file(custom_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")


def generate_and_display_image(prompt, label, num_steps, sampler, temperature, guidance_scale, width, height, super_resolution):
    def generate():
        global pil_image
        # Set the number of inference steps for the model
        # pipe.set_config(num_steps=num_steps, sampler=sampler, temperature=temperature, ...)
        pipe.num_inference_steps = num_steps

        with torch.no_grad():
            image = pipe(prompt=prompt,num_inference_steps=num_steps).images[0]
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        pil_image = image.resize((width, height))  # Resize the image
        tk_image = ImageTk.PhotoImage(pil_image)
        label.config(image=tk_image)
        label.image = tk_image  # Keep a reference

    threading.Thread(target=generate).start()

def on_enter_pressed():
    selected_model = model_selector.get()
    custom_path = custom_model_path_input.get() if selected_model == "Custom Path" else None
    initialize_model(selected_model, custom_path)
    positive_prompt = text_input.get()
    negative_prompt = negative_text_input.get()
    combined_prompt = f"{positive_prompt}\nNegative prompt: {negative_prompt}"  # Combine prompts
    num_steps = int(num_steps_input.get())
    sampler = sampler_input.get()
    temperature = temperature_input.get()
    guidance_scale = guidance_scale_input.get()
    width = int(width_input.get())
    height = int(height_input.get())
    super_resolution = bool(super_resolution_var.get())
    generate_and_display_image(combined_prompt, label, num_steps, sampler, temperature, guidance_scale, width, height, super_resolution)
    
def save_generated_image():
    if pil_image is not None:
        save_image(pil_image)
    else:
        print("No image to save")
    
    
    
style = Style(theme='darkly')
root = style.master
root.title("bhanuyash's Image Generator")
root.geometry("1024x768")  

# Parameter frame
param_frame = ttk.Frame(root)
param_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# Prompt frame
prompt_frame = ttk.Frame(root)
prompt_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Parameters with default values
default_num_steps = 50
default_sampler = "DDIM"
default_temperature = 1.0
default_guidance_scale = 7.5
default_width = 512
default_height = 512


# Add model selection dropdown
ttk.Label(param_frame, text="Select Model").grid(row=7, column=0, sticky="w")
model_selector = ttk.Combobox(param_frame, values=["Stable Diffusion", "DALL-E", "DreamShaper XL", "Custom Path"], state="readonly")
model_selector.grid(row=7, column=1, pady=5)
model_selector.set("Stable Diffusion")  # Default selection

# Add custom path entry
ttk.Label(param_frame, text="Custom Model Path").grid(row=8, column=0, sticky="w")
custom_model_path_input = ttk.Entry(param_frame, width=60)
custom_model_path_input.grid(row=8, column=1, pady=5)

# Parameter inputs
ttk.Label(param_frame, text="Number of Steps").grid(row=0, column=0, sticky="w")
num_steps_input = ttk.Entry(param_frame, width=10)
num_steps_input.grid(row=0, column=1, pady=5)
num_steps_input.insert(0, str(default_num_steps))

ttk.Label(param_frame, text="Sampler").grid(row=1, column=0, sticky="w")
samplers = ["DDIM", "DDPM", "Euler a", "DPM++ 2S a Karras", "DPM++ 3M SDE Exponential", "DPM++ 2M SDE Karras"]
sampler_input = ttk.Combobox(param_frame, values=samplers, state="readonly")
sampler_input.grid(row=1, column=1, pady=5)
sampler_input.set(default_sampler)

ttk.Label(param_frame, text="Temperature").grid(row=2, column=0, sticky="w")
temperature_input = ttk.Scale(param_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
temperature_input.grid(row=2, column=1, pady=5)
temperature_input.set(default_temperature)

ttk.Label(param_frame, text="Guidance Scale").grid(row=3, column=0, sticky="w")
guidance_scale_input = ttk.Scale(param_frame, from_=1.0, to=20.0, orient=tk.HORIZONTAL)
guidance_scale_input.grid(row=3, column=1, pady=5)
guidance_scale_input.set(default_guidance_scale)

ttk.Label(param_frame, text="Width").grid(row=4, column=0, sticky="w")
width_input = ttk.Entry(param_frame, width=10)
width_input.grid(row=4, column=1, pady=5)
width_input.insert(0, str(default_width))

ttk.Label(param_frame, text="Height").grid(row=5, column=0, sticky="w")
height_input = ttk.Entry(param_frame, width=10)
height_input.grid(row=5, column=1, pady=5)
height_input.insert(0, str(default_height))

super_resolution_var = tk.IntVar(value=0)
super_resolution_check = ttk.Checkbutton(param_frame, text="Super Resolution", variable=super_resolution_var)
super_resolution_check.grid(row=6, column=0, columnspan=2, pady=5)

# Negative prompt frame
negative_prompt_frame = ttk.Frame(root)
negative_prompt_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
ttk.Label(negative_prompt_frame, text="Enter Negative Prompt:").pack(side=tk.LEFT)
negative_text_input = ttk.Entry(negative_prompt_frame, width=60)
negative_text_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

# Generate button frame
generate_button_frame = ttk.Frame(root)
generate_button_frame.pack(side=tk.BOTTOM, anchor='e', padx=10, pady=10)

# Modify the generate button command
generate_button = ttk.Button(generate_button_frame, text="Generate", command=on_enter_pressed)
generate_button.pack(side=tk.RIGHT, anchor='e', pady=10)

# Prompt entry
ttk.Label(prompt_frame, text="Enter Prompt:").pack(side=tk.LEFT)
text_input = ttk.Entry(prompt_frame, width=60)
text_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

label = ttk.Label(root)
label.pack(pady=10)

pil_image = None

save_button = ttk.Button(root, text="Save Image", command=save_generated_image)
save_button.pack(pady=10)

# Bind the event handler
text_input.bind("<Return>", lambda event: on_enter_pressed())

text_widget = tk.Text(root, height=10, width=50, bg="black", fg="white")
text_widget.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=10, padx=10)

sys.stdout = ConsoleRedirector(text_widget)
sys.stderr = ConsoleRedirector(text_widget)

root.mainloop()