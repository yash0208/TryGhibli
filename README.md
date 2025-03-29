# TryGhibli - Transform Your Images into Ghibli-Style Art 🎨✨

## Introduction
Want to turn your favorite memories into **Studio Ghibli-style** masterpieces? With **Stable Diffusion’s img2img pipeline**, you can achieve stunning anime-style transformations using an open-source model from **Hugging Face**.

This repository provides a simple Python script to convert any image into a **Ghibli-inspired painting** using **Stable Diffusion 2**.

---

## Features 🚀
✅ **AI-Powered Transformation** - Uses **Stable Diffusion 2** for high-quality image-to-image conversion.  
✅ **Ghibli-Style Art** - Applies a dreamy, colorful, anime aesthetic.  
✅ **Easy to Use** - Just run the script with an input image, and you’re done!  
✅ **Runs on GPU** - Supports CUDA for faster processing.  

---

## Installation 🔧
1. **Clone the Repository**
```sh
git clone https://github.com/yash0208/TryGhibli.git
cd TryGhibli
```

2. **Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. **Install Dependencies**
```sh
pip install torch diffusers pillow
```

---

## Usage 🖼️
1. **Run the Python Script**
```sh
python ghibli_style.py input.jpg
```

2. **Example Code** (Included in `ghibli_style.py`)
```python
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def ghibli_style_transform(image_path, output_path="ghibli_output.png"):
    model_id = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    prompt = "A Ghibli-style painting, vibrant colors, dreamy background, anime aesthetic"
    result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    
    result.save(output_path)
    print(f"Transformed image saved as {output_path}")
```

---
