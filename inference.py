import json
import torch
import torch.nn.functional as F
from model import ScalableDecoderOnly

model_name = "./model_logs/josh_model_epoch_10.pt"

# === Load config and token mappings ===
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open("tokens.json", "r", encoding="utf-8") as f:
    token_to_id = json.load(f)
id_to_token = {v: k for k, v in token_to_id.items()}

# === Model setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ScalableDecoderOnly().to(device)

model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()

pad_token = token_to_id["<pad>"]
eos_token = token_to_id["<eos>"]
unk_token = token_to_id["<unk>"]

# === Inference Function ===
def generate_response(prompt, max_len=config.get("max_len", 100)):
    tokens = prompt.lower().strip().split()
    input_ids = [token_to_id.get(tok, unk_token) for tok in tokens] + [eos_token]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = input_ids.copy()

    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor)
            next_token_logits = output[0, -1]
            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1)).item()

        if next_token == eos_token or (len(generated) > 2 and next_token == generated[-1]):
            break

        generated.append(next_token)
        input_tensor = torch.tensor([generated], dtype=torch.long).to(device)

    response = [id_to_token.get(tok, "<unk>") for tok in generated[len(input_ids):]]
    return " ".join(response)

# === Loop for chatting ===
if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip().lower()
        if user_input in ["exit", "quit"]:
            break
        response = generate_response(user_input)
        print("Josh:", response)
