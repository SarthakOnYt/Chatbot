import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load the model and tokenizer locally
def load_model():
    model_path = "./Mistral-7B"  # Path to the cloned Git repository

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # Automatically chooses the best device
            offload_folder="./offload",  # Saves offloaded weights to disk
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use float16 on GPU for speed
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model and tokenizer loaded successfully!")

        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

# Initialize ChromaDB for storing conversation history
def init_chroma():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db

# Function to store conversations in ChromaDB
def store_memory(db, user_input, ai_response):
    try:
        db.add_texts([f"User: {user_input}\nAI: {ai_response}"])
        db.persist()
        print("✅ Conversation stored in memory.")
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")

# Retrieve memory for context
def retrieve_memory(db, query, k=3):
    try:
        results = db.similarity_search(query, k=k)
        return [result.page_content for result in results]
    except Exception as e:
        print(f"❌ Error retrieving memory: {e}")
        return []

# Generate AI response using the model
def generate_response(model, tokenizer, prompt):
    # Tokenize the input prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id).long()

    # Ensure to place inputs on the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate the model's response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_length=100,  # You can adjust this depending on desired response length
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Handle padding properly
        )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure we only return the last generated portion (the AI's response)
    response = response.split("AI:")[-1].strip()
    
    return response

# Main conversation loop
def ai_vtuber():
    model, tokenizer = load_model()
    db = init_chroma()
    conversation_history = []  # List to store the entire conversation

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break

            # Add the user's input to the conversation history
            conversation_history.append(f"User: {user_input}")

            # Construct the full conversation context
            conversation_context = "\n".join(conversation_history)  # All prior conversation history
            prompt = conversation_context + "\nAI:"  # Adding "AI:" for the model to respond to

            # Generate AI response
            ai_response = generate_response(model, tokenizer, prompt)
            print(f"AI: {ai_response}")

            # Store the conversation in memory (both user and AI response)
            store_memory(db, user_input, ai_response)

            # Add the AI's response to the conversation history
            conversation_history.append(f"AI: {ai_response}")

        except KeyboardInterrupt:
            print("\n👋 Exiting AI VTuber. Goodbye!")
            break

# Start the chatbot
if __name__ == "__main__":
    ai_vtuber()
