# AI Art Director 🎨

An intelligent web application that transforms simple user ideas into complex, high-quality images using a two-step AI pipeline. This project demonstrates model chaining, prompt engineering, and building user-facing AI applications.




---

## The Core Concept: A Two-Step AI Team

Many users struggle to write detailed prompts for AI image generators. This application solves that problem by automating the creative process with an AI "team":

1.  **The Creative Director (Prompt Generation):** The user provides a simple idea (e.g., "a cat in a library"). The application feeds this idea to an instruction-following Large Language Model (**Google's Flan-T5**). The Director's job is to brainstorm and write a rich, artistic, and detailed prompt that describes the scene with a focus on lighting, style, and mood.

2.  **The Artist (Image Generation):** The detailed prompt from the Director is then passed to a powerful diffusion model (**Stable Diffusion v1.5**). The Artist uses this high-quality instruction to generate the final, beautiful piece of art.

### From Bug to Feature: Why Flan-T5 is Crucial

The project initially used `distilgpt2` as the Creative Director. However, it often failed to follow instructions, producing nonsensical prompts (e.g., turning "a small mouse sleeping" into "THE ONLY DAD FOR FAN").

This demonstrated a key challenge in AI engineering: choosing the right model for the task.

The model was upgraded to `google/flan-t5-base`, an instruction-tuned model. This move dramatically improved the reliability and quality of the generated prompts, turning a critical bug into a core feature: **reliable, high-quality, AI-driven prompt engineering.**

---

## Key Features

*   **AI Pipeline (Model Chaining):** Leverages two distinct AI models working in sequence to achieve a complex task.
*   **Instruction-Following LLM:** Uses Flan-T5 for reliable and context-aware prompt generation.
*   **Advanced User Controls:** The Streamlit interface provides controls for art style, negative prompts, and guidance scale (CFG).
*   **Performance Optimized:** Utilizes Streamlit's `@st.cache_resource` to load the large AI models only once, ensuring a fast and smooth user experience after the initial startup.
*   **Full-Stack Application:** A complete, interactive web application built with a Python backend (PyTorch, Transformers) and a Streamlit frontend.

---

## Tech Stack

*   **Language:** Python
*   **Web Framework:** Streamlit
*   **AI/ML Libraries:** PyTorch, Hugging Face `Transformers`, `Diffusers`, `accelerate`
*   **Models:**
    *   `google/flan-t5-base` (Language Model)
    *   `runwayml/stable-diffusion-v1-5` (Diffusion Model)

---

## Local Setup and Installation

To run this project on your own machine, follow these steps.

**Prerequisites:**
*   Python 3.8+
*   Git

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repository-link]
    cd ai-art-director
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

**NOTE:** The first time you run the app, it will download the AI models, which can be several gigabytes. Image generation will be very slow on a CPU. A CUDA-enabled NVIDIA GPU or an Apple Silicon Mac is highly recommended for reasonable performance.
