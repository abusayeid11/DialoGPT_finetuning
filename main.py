#https://drive.google.com/file/d/1P6S2Gia7YzHSPfGoXIFmlc8O91FGcjkl


from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gdown
import os
import zipfile

app = Flask(__name__)
CORS(app)  # Enable CORS

# Google Drive direct download link (replace with your file's shareable link)
DRIVE_MODEL_ZIP = "https://drive.google.com/uc?id=1P6S2Gia7YzHSPfGoXIFmlc8O91FGcjkl"  # 👈 Replace with your Drive file ID
MODEL_DIR = "./fine_tuned_dialogpt_career_advisor"  # Where the unzipped model will be stored

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading model from Google Drive...")
        #Download the zip
        gdown.download(DRIVE_MODEL_ZIP, "model.zip", quiet=False)
        # Unzip
       # Replace unzip command with this:
        print("Extracting model...")
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall("./")  # Extracts to current directory
        os.remove("model.zip")  # Optional: delete zip after extraction
        print("Model extracted!")

# Initialize model (load only once)
print("Loading model...")
download_and_extract_model()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
print("Model loaded!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    max_length = data.get('max_length', 150)

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Format input if your model expects special tokens
    formatted_input = f"USER: {message}\nASSISTANT:"
    
    inputs = tokenizer.encode(formatted_input, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    # Extract only the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = full_response.split("ASSISTANT:")[-1].strip()
    
    return jsonify({"response": assistant_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)