from diffusers import StableDiffusionXLPipeline
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

def save_image(image, path='Image-generator-stable-diffusion/Images'):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(os.path.join(path, "generated_image.png"))

def generate_and_display_image(prompt, label, num_steps, sampler, temperature, guidance_scale, width, height, super_resolution):
    def generate():
        global pil_image
        with torch.no_grad():
            # Modify here to use the additional parameters
            image = pipe(prompt=prompt).images[0]
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        pil_image = image.copy()
        tk_image = ImageTk.PhotoImage(image)
        label.config(image=tk_image)

    threading.Thread(target=generate).start()

def on_enter_pressed(event, label):
    prompt = text_input.get()
    num_steps = int(num_steps_input.get())
    sampler = sampler_input.get()
    temperature = temperature_input.get()
    guidance_scale = guidance_scale_input.get()
    width = int(width_input.get())
    height = int(height_input.get())
    super_resolution = bool(super_resolution_var.get())
    generate_and_display_image(prompt, label, num_steps, sampler, temperature, guidance_scale, width, height, super_resolution)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

style = Style(theme='darkly')
root = style.master
root.title("bhanuyash's Image Generator")
root.geometry("800x600")  # Increased UI size

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

# Parameter inputs
ttk.Label(param_frame, text="Number of Steps").grid(row=0, column=0, sticky="w")
num_steps_input = ttk.Entry(param_frame, width=10)
num_steps_input.grid(row=0, column=1, pady=5)
num_steps_input.insert(0, str(default_num_steps))

ttk.Label(param_frame, text="Sampler").grid(row=1, column=0, sticky="w")
sampler_input = ttk.Combobox(param_frame, values=["DDIM", "DDPM"], state="readonly")
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

# Prompt entry
ttk.Label(prompt_frame, text="Enter Prompt:").pack(side=tk.LEFT)
text_input = ttk.Entry(prompt_frame, width=60)
text_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

label = ttk.Label(root)
label.pack(pady=10)

pil_image = None

def save_generated_image():
    if pil_image is not None:
        save_image(pil_image)
    else:
        print("No image to save")

save_button = ttk.Button(root, text="Save Image", command=save_generated_image)
save_button.pack(pady=10)

text_input.bind("<Return>", lambda event, lbl=label: on_enter_pressed(event, lbl))

text_widget = tk.Text(root, height=10)
text_widget.pack(pady=10)
sys.stdout = ConsoleRedirector(text_widget)

root.mainloop()