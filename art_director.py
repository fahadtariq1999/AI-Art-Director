# art_director.py (Version 3 - Better Prompting)

# This line can help hide the harmless TensorFlow warning if you have TensorFlow installed.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# --- 1. Load the AI Models ---

# Load the "Creative Director" (a text generation model)
print("Loading the Creative Director (GPT-2)...")
director = pipeline('text-generation', model='distilgpt2')

# Load the "Artist" (a diffusion model)
print("Loading the Artist (Stable Diffusion)...")
model_id = "runwayml/stable-diffusion-v1-5"
artist = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move model to GPU if available
if torch.cuda.is_available():
    print("GPU detected! Moving artist model to CUDA.")
    artist = artist.to("cuda")
elif torch.backends.mps.is_available():
    print("Apple Silicon GPU detected! Moving artist model to MPS.")
    artist = artist.to("mps")
else:
    print("WARNING: No GPU detected. The artist will run on the CPU, which will be very slow (5-30 minutes per image).")
    artist = artist.to("cpu")

artist.enable_attention_slicing()


# --- 2. Define the Main Function ---

def create_art(simple_idea: str):
    """
    This function takes a simple idea and generates a piece of art.
    """
    print(f"Received simple idea: '{simple_idea}'")

    # --- Step A: THINK (Chain of Thought / Reasoning) ---
    # --- NEW: SIMPLER AND MORE DIRECT PROMPT TEMPLATE ---
    # This template gives a clear instruction instead of an example.
    prompt_template = f"""
Take the following simple idea and turn it into a rich, artistic, and descriptive prompt for an image generator. Focus on visual details like lighting, style, and mood. Only output the detailed prompt.

SIMPLE IDEA: "{simple_idea}"

DETAILED PROMPT:
"""
    
    print("Creative Director is brainstorming a detailed prompt...")
    director_output = director(prompt_template, max_new_tokens=60, num_return_sequences=1)
    
    # --- NEW: UPDATED PARSING LOGIC FOR THE NEW TEMPLATE ---
    full_text = director_output[0]['generated_text']
    # This splits the output at our keyword and takes everything that comes after it.
    try:
        detailed_prompt = full_text.split('DETAILED PROMPT:')[1].strip()
    except IndexError:
        # Fallback in case the model doesn't follow instructions perfectly
        detailed_prompt = simple_idea + ", beautiful digital art"


    print(f"Detailed prompt created: '{detailed_prompt}'")

    # --- Step B: ACT (Action) ---
    print("Artist is now creating the image... Please be patient.")
    # The 'num_inference_steps' can be lowered to speed up generation on a CPU, at a slight cost to quality.
    # Default is 50. Let's use 20 for faster testing.
    image = artist(detailed_prompt, num_inference_steps=20).images[0]
    
    image_filename = f"{simple_idea.replace(' ', '_')}.png"
    image.save(image_filename)
    
    print(f"Success! Your art has been saved as '{image_filename}'")
    return image, detailed_prompt, image_filename


# --- 3. Run the project ---
if __name__ == '__main__':
    # This code only runs when you execute this file directly
    user_idea = "a robot reading a book in a forest"
    create_art(user_idea)