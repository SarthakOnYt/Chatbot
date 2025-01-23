import torch
import json
from functools import lru_cache
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Context setup for bot
Context = {
    "Name": "", # Name of the Ai model
    "Age": "You are an AI and hence do not have an age", # Age of the AI
    "Role": "",# Role of the AI
    "Allowed": ["Family friendly content","Jokes"],
    "NotAllowed": ["Advertising", "Spamming", "Hate speech"]
}

# Chat History Management
def load_history():
    try:
        with open("chat_history.json", "r") as ri:
            return json.load(ri)
    except FileNotFoundError:
        return []  # Return an empty list if no file is found

def save_history(chat_history):
    with open("chat_history.json", "w") as wi:
        json.dump(chat_history, wi, indent=4)

# Initialize chat history
chat_history = load_history()

formatted_chat_history=""
for a in chat_history:
    formatted_chat_history += f"Role: {a['role']}\nContent: {a['content']}\n\n"

print("Chat History Loaded")
#----
model_path = "./Mistral-7B"  # Path to Model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatically chooses the best device
    offload_folder="./offload",  # Saves offloaded weights to disk
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use float16 on GPU for speed
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")


#Use pipeline is device is of mid-high end
"""
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    #device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)"""

#Remove the tag if you want to use caching for generated responses
#@lru_cache(maxsize=1000) 
def generate_response(Text_input):
    #model = text_generation_pipeline
    
    Full_input = (
    f"Context: {Context}\n"
    f"Chat History:\n{formatted_chat_history}\n"
    f"User: {Text_input}\n")

    # Tokenize input with padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        Full_input, 
        return_tensors="pt", 
        padding=True,   # Adds padding to match length
        truncation=True,  # Truncates if too long
        max_length=200  # Set max length to avoid unexpected behaviors
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention = inputs["attention_mask"].to(model.device)

    response = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        attention_mask=attention,
        pad_token_id=tokenizer.eos_token_id
        )
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    
    chat_history.append({"role": "user", "content": Text_input})
    chat_history.append({"role": "Ai", "content": response_text})
    save_history(chat_history)

    return response_text

# Chat interaction loop
def chat():
    while True:
        Text_input = input("User: ")
        if Text_input.lower() == 'q':
            break
        response = generate_response(Text_input)
        print(f"Josh: {response}")

chat()
