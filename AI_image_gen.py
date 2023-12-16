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

def generate_and_display_image(prompt, label):
    def generate():
        global pil_image
        with torch.no_grad():
            image = pipe(prompt=prompt).images[0]
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        pil_image = image.copy()
        tk_image = ImageTk.PhotoImage(image)
        label.config(image=tk_image)
        label.image = tk_image

    threading.Thread(target=generate).start()

def on_enter_pressed(event, label):
    prompt = text_input.get()
    generate_and_display_image(prompt, label)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

style = Style(theme='darkly')
root = style.master
root.title("bhanuyash's Image Generator")

text_input = ttk.Entry(root, width=60)
text_input.pack(pady=10)

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
text_widget.pack()
sys.stdout = ConsoleRedirector(text_widget)

root.mainloop()