# Career Guide Chatbot Fine-tuning with LoRA
# Optimized for Google Colab

# Install required packages
!pip install -q transformers datasets peft accelerate bitsandbytes trl torch pandas

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import json

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Alternative: "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./career-guide-chatbot"
HF_TOKEN = "your_huggingface_token_here"  # Optional, for private models

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank - can adjust (8, 16, 32)
    lora_alpha=32,  # Alpha parameter
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def load_and_prepare_data(csv_file_path):
    """
    Load CSV data and convert to training format
    Expected CSV format: role, content_topic
    """
    df = pd.read_csv(csv_file_path)
    
    # Create conversation format
    formatted_data = []
    
    for _, row in df.iterrows():
        role = row['role']
        content = row['content_topic']
        
        # Create a career guidance conversation format
        if 'software engineer' in role.lower():
            prompt = f"I want to become a {role}. Can you guide me?"
        elif 'machine learning' in role.lower():
            prompt = f"What should I know to pursue a career in {role}?"
        else:
            prompt = f"How can I start a career as a {role}?"
        
        # Format as instruction-response pair
        conversation = f"""<|system|>
You are a career guidance counselor specializing in computer science and technology careers. Provide helpful, practical advice for career development.<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>
{content}<|end|>"""
        
        formatted_data.append({"text": conversation})
    
    return Dataset.from_list(formatted_data)

# Load your data
# Replace 'your_data.csv' with your actual CSV file path
train_dataset = load_and_prepare_data('your_data.csv')

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small batch size for Colab
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision training
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",  # No validation for simplicity
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None,  # Disable wandb logging
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
)

# Start training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

# Save LoRA adapters specifically
model.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")

# Testing the trained model
def test_model():
    """Test the fine-tuned model"""
    # Load the fine-tuned model for inference
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    
    # Test questions
    test_prompts = [
        "<|system|>You are a career guidance counselor specializing in computer science and technology careers. Provide helpful, practical advice for career development.<|end|><|user|>I want to become a software engineer. What skills should I focus on?<|end|><|assistant|>",
        "<|system|>You are a career guidance counselor specializing in computer science and technology careers. Provide helpful, practical advice for career development.<|end|><|user|>How do I start a career in machine learning?<|end|><|assistant|>"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt.split('<|user|>')[1].split('<|end|>')[0]}")
        response = pipe(prompt)
        print(f"Response: {response[0]['generated_text'][len(prompt):]}")

# Uncomment to test after training
# test_model()

# Download files to local device (for Colab)
def download_model():
    """Package and download the trained model"""
    import shutil
    
    # Create a zip file of the model
    shutil.make_archive('career_guide_chatbot', 'zip', OUTPUT_DIR)
    
    # In Colab, this will trigger download
    from google.colab import files
    files.download('career_guide_chatbot.zip')

# Uncomment to download after training
# download_model()

# Usage example for inference
"""
After downloading, you can use the model like this:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Load your fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, "./path/to/your/downloaded/model")

# Create pipeline for easy inference
from transformers import pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Use the chatbot
response = chatbot("I want to become a data scientist. What should I do?")
"""