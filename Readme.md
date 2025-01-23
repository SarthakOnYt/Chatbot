# Chatbot information

This Python script demonstrates a simple chatbot using the large language models provided on huggingface, simply replace modelpath with your desired path.

# Tested with

1. **PYTHIA:12B**
2. **TinyLlama:**
3. **Mistral:7B**

## Features

-   **Model Loading:**
    -   Loads any model using the `transformers` library.
    -   Utilizes `device_map="auto"` for efficient resource allocation.
    -   Employs `torch_dtype=torch.float16` on GPUs for faster inference.
-   **Contextual Conversation:**
    -   Maintains a chat history in a JSON file (`chat_history.json`).
    -   Includes basic context information about the AI (Name, Age, Role, Allowed, NotAllowed).
-   **Text Generation:**
    -   Uses the model to generate responses based on user input, context, and chat history.
    -   Provides options for adjusting parameters like `temperature` and `top_p` to control the creativity and randomness of the generated text.
-   **User Interaction:**
    -   Allows users to interact with the chatbot through a simple command-line interface.

## Requirements

-   Python 3.7 or higher
-   `torch`
-   `transformers`

## Installation

1. **Install required libraries:**
    ```bash
    pip install torch transformers
    ```

## Current Known issues

1. **Bot starts a conversation with itself and then sends a whole convrsation on something in the reply:**
