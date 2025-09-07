# Multi-Modal AI Assistant

This project is a **state-of-the-art multi-modal AI assistant** that integrates **text generation, image generation, and knowledge retrieval** into a single interactive platform. It combines advanced **TextGeneration models**, **LoRA-fine-tuned image generation**, and **Retrieval-Augmented Generation (RAG)** techniques to deliver a seamless and intelligent user experience.

---

## Project Structure

The project contains **two main Python files**:

1. **`Multi_model_Gen_AI.py`**  
   - Responsible for **training and saving models**.  
   - Includes:
     - Text generation model setup, training, and LoRA fine-tuning.
     - Image generation fine-tuning using LoRA.
     - RAG pipeline setup for knowledge retrieval.
   - Saves the trained/fine-tuned models to a specified directory.

2. **`Multi_model_Gen_AI2.py`**  
   - Provides a **Streamlit-based interactive dashboard** for real-time AI assistant usage.
   - Features:
     - Text input for user queries.
     - Automatic text generation response.
     - Image generation from prompts using the LoRA model.
     - RAG-powered context-aware answers.
     - Persistent chat history with text and images.
   - **Important:** Users must provide their **own saved model directory**. Replace the blank variable `save_dir = "........."` with the path where your models are stored.

---

## Key Features

- **Text Generation:** context-aware answers using advanced models.  
- **Image Generation:** Image creation via LoRA fine-tuning.  
- **Retrieval-Augmented Generation (RAG):** Provides accurate, knowledge-driven responses from external documents or datasets.  
- **Interactive Dashboard:** Streamlit interface for real-time chat and image display.  
- **Persistent Conversation History:** Maintains a smooth chat experience with session-based history.

---

## How to Use

1. **Train and Save Models**
   ```bash
   python Multi_model_Gen_AI.py
   Configure the save directory in the script before running.

2. Run the AI Assistant

streamlit run Multi_model_Gen_AI2.py


Open your browser at the provided Streamlit URL.

Replace the save_dir = "........." variable with your saved model directory.

Type queries in the "Ask me anything" box or use prompts like generate image of ....

Press Enter or click Submit to interact.
