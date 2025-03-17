from flask import Flask, request, jsonify, render_template
import os
import requests

app = Flask(__name__)

# Load API Key securely
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("❌ API Key not found! Check your environment variables.")

# Hugging Face Model Details
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

import re

def clean_response(response):
    """Remove system tokens like [INST] and [/INST] from the chatbot's response"""
    response = re.sub(r'\[INST\]|\[/INST\]', '', response)  # Remove [INST] and [/INST]
    return response.strip()  # Remove any extra spaces or newlines

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")
    bot_response = generate_response(user_input)  # Call your function to get AI's response
    clean_bot_response = clean_response(bot_response)  # Clean up response
    return jsonify({"response": clean_bot_response})


# Function to call Hugging Face API
def ask_supermom(question):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "inputs": f"<s>[INST] {SYSTEM_MESSAGE}\n\nUser: {question} [/INST]>",
        "parameters": {"max_new_tokens": 200, "temperature": 0.7, "top_p": 0.9},
    }
    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"].strip() if isinstance(result, list) else "Unexpected response format."
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network Error: {str(e)}"

# 🔹 Serve HTML Page
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 API Endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = ask_supermom(question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
