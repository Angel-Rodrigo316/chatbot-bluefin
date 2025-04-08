from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

history = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id", "anon")
    message = data.get("message", "")

    if user_id not in history:
        history[user_id] = None

    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([history[user_id], input_ids], dim=-1) if history[user_id] is not None else input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    history[user_id] = chat_history_ids

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render asigna automáticamente un puerto
    app.run(host="0.0.0.0", port=port)