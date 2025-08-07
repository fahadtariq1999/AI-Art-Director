import streamlit as st
import torch

# --- UI Configuration ---
st.set_page_config(page_title="AI Art Director", layout="wide")
st.title("🎨 AI Art Director ")
st.write("A more reliable AI team. The Director now uses a superior instruction-following model (Flan-T5) to create better prompts.")

# --- Model Caching ---
@st.cache_resource
def load_models():
    from transformers import pipeline
    from diffusers import StableDiffusionPipeline
    import torch

    st.write("Booting up the AI team... This will take a moment.")
    
    # --- NEW: Use a more capable instruction-following model for the Director ---
    # Flan-T5 is much better at following our prompt instructions than distilgpt2.
    # The task is also different: 'text2text-generation'.
    st.write("Loading Creative Director (Flan-T5)...")
    director_model_name = "google/flan-t5-base" # Switched from distilgpt2
    director = pipeline('text2text-generation', model=director_model_name, torch_dtype=torch.bfloat16)

    # Load the "Artist" (a diffusion model)
    st.write("Loading Artist (Stable Diffusion)...")
    artist_model_name = "runwayml/stable-diffusion-v1-5"
    artist = StableDiffusionPipeline.from_pretrained(artist_model_name, torch_dtype=torch.float16)

    # Move models to GPU if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        st.write("GPU detected! Moving models to CUDA.")
    elif torch.backends.mps.is_available():
        device = "mps"
        st.write("Apple Silicon GPU detected! Moving models to MPS.")
    else:
        st.write("WARNING: No GPU detected. The artist will run on the CPU, which may be slow.")
    
    artist = artist.to(device)
    # Flan-T5 will also be moved to the device by the pipeline automatically.
    
    artist.enable_attention_slicing()
    st.success("AI team is ready!")
    return director, artist

director_model, artist_model = load_models()

# --- Slightly modified art creation function ---
def create_art_v3(simple_idea: str, style_modifier: str, negative_prompt: str, guidance_scale: float):
    """
    Generates art using the upgraded Flan-T5 Director model.
    """
    print(f"Received simple idea: '{simple_idea}'")

    # Construct the final idea with the style modifier
    final_idea = f"{simple_idea}, in the style of {style_modifier}"

    # --- NEW: A simpler, more direct prompt template for Flan-T5 ---
    prompt_template = f"""
Create a rich, beautiful, and highly detailed image prompt for an AI art generator.
The prompt should be based on the following idea: "{final_idea}"
"""
    
    print("Creative Director (Flan-T5) is brainstorming a detailed prompt...")
    # The output format for text2text models is slightly different
    director_output = director_model(prompt_template, max_new_tokens=1024, do_sample=False)
    detailed_prompt = director_output[0]['generated_text'].strip()

    print(f"Detailed prompt created: '{detailed_prompt}'")
    
    # --- Fallback and Cleanup ---
    # Sometimes the model might add extra quotes or prefixes. Let's clean it.
    if detailed_prompt.startswith('"') and detailed_prompt.endswith('"'):
        detailed_prompt = detailed_prompt[1:-1]
    if not detailed_prompt: # If the prompt is empty for some reason
        print("Director failed, using fallback prompt.")
        detailed_prompt = final_idea + ", beautiful digital art"


    print("Artist is now creating the image... Please be patient.")
    image = artist_model(
        prompt=detailed_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=guidance_scale
    ).images[0]
    
    image_filename = f"{simple_idea.replace(' ', '_')}.png"
    image.save(image_filename)
    
    print(f"Success! Your art has been saved as '{image_filename}'")
    return image, detailed_prompt, image_filename

# --- UI Elements (same as before) ---
with st.sidebar:
    st.header("🎨 Art Controls")
    
    style_choice = st.selectbox(
        "Choose an art style:",
        ("cinematic lighting", "vibrant color palette", "photorealistic", "impressionist painting", "cyberpunk aesthetic", "fantasy concept art", "minimalist line art")
    )

    guidance_slider = st.slider(
        "Artistic Guidance (CFG Scale)", 
        min_value=7.0, 
        max_value=15.0, 
        value=7.5, 
        step=0.5,
        help="How strictly the artist follows the prompt. Higher values are more strict but can be less creative."
    )
    
    negative_input = st.text_input(
        "Negative Prompt (what to avoid):", 
        "blurry, watermark, text, deformed, ugly, bad anatomy"
    )

with st.form("idea_form"):
    user_idea = st.text_input("Enter your simple art idea:", "A small mouse sleeping in a cozy burrow")
    submitted = st.form_submit_button("Create Art!")

# --- Generate and Display Art ---
if submitted:
    if user_idea:
        with st.spinner("The AI team is at work... This may take a moment."):
            try:
                generated_image, detailed_prompt, image_filename = create_art_v3(
                    user_idea, 
                    style_modifier=style_choice,
                    negative_prompt=negative_input,
                    guidance_scale=guidance_slider
                )

                st.success("Art created successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(generated_image, caption="Final Artwork", use_column_width=True)
                with col2:
                    st.subheader("The Director's Brainstorm (Flan-T5)")
                    st.info(detailed_prompt)

                with open(image_filename, "rb") as file:
                    st.download_button(
                        label="Download Image",
                        data=file,
                        file_name=image_filename,
                        mime="image/png"
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e) # This will print the full traceback for debugging
    else:
        st.warning("Please enter an idea.")