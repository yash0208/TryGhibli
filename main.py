import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

def ghibli_style_transform(image_path, output_path="ghibli_output.png"):
    # Load the model from Hugging Face
    model_id = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the image
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    
    # Define the transformation prompt
    prompt = "A Ghibli-style painting, vibrant colors, dreamy background, anime aesthetic"
    
    # Generate the transformed image
    result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    
    # Save the output image
    result.save(output_path)
    print(f"Transformed image saved as {output_path}")

# Example usage
ghibli_style_transform("input.jpg")
