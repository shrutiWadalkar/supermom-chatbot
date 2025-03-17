import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("❌ API Key not found! Set it in Render's environment variables.")

# ✅ AI Model & System Message
model_name = "HuggingFaceH4/zephyr-7b-beta"
system_message = """You are SuperMom Guide, an AI parenting assistant.
Your goal is to provide clear, practical, and step-by-step parenting advice.
You will give only **real-world parenting solutions** with no fiction.
"""

# ✅ Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ SuperMom AI is Running! Use /ask endpoint."

@app.route("/ask", methods=["POST"])
def ask_supermom():
    """Receive a question and return AI response."""
    data = request.json
    question = data.get("question", "")

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "inputs": f"<s>[INST] {system_message}\n\nUser: {question} [/INST]>",
        "parameters": {"max_new_tokens": 200, "temperature": 0.7, "top_p": 0.9},
    }

    try:
        response = requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return jsonify({"response": result[0]["generated_text"].strip() if isinstance(result, list) else "Error: Unexpected format"})

        return jsonify({"error": f"API Error: {response.status_code} - {response.text}"})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # ✅ Runs on Render
