# finetune_dialog.py - CORRECTED for your specific files

import pandas as pd
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os
from glob import glob

# --- Configuration ---
# Path to your new CSV file within the knowledge_base directory
NEW_CSV_PATH = 'knowledge_base/datascienceQnA.csv'
# Directory containing your existing JSON knowledge base files (correctly set to the folder)
KNOWLEDGE_BASE_DIR = 'knowledge_base'
# Directory where the fine-tuned model will be saved
FINE_TUNED_MODEL_OUTPUT_DIR = "./fine_tuned_dialogpt_career_advisor"

# --- 1. Initialize Tokenizer EARLY ---
print("--- Initializing Tokenizer (early for data prep) ---")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# Add a padding token if it doesn't have one (DialoGPT often uses eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized.")

# --- 2. Data Loading and Preparation ---
print("\n--- Loading and Preparing Data ---")
combined_conversations = []

# A. Load conversations from existing knowledge base JSONs
print(f"Loading data from knowledge base JSONs in directory: {KNOWLEDGE_BASE_DIR}")
json_files = glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.json"))
if not json_files:
    print(f"Warning: No JSON files found in '{KNOWLEDGE_BASE_DIR}'. Check path or ensure files exist.")

for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            topic_data = json.load(f)
            # Use os.path.basename to get just the filename, then split and replace underscores
            topic_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()

            # Generate conversational pairs from your JSON structure
            if "overview" in topic_data:
                combined_conversations.append(f"User: Tell me about {topic_name}. Bot: {topic_data['overview']}{tokenizer.eos_token}")
                combined_conversations.append(f"User: What is {topic_name}? Bot: {topic_data['overview']}{tokenizer.eos_token}")

            if "pros" in topic_data and isinstance(topic_data["pros"], list):
                # Using a more compact string for training examples
                pros_str = " • ".join(topic_data["pros"])
                combined_conversations.append(f"User: What are the pros of {topic_name}? Bot: Some advantages include: {pros_str}{tokenizer.eos_token}")
            if "cons" in topic_data and isinstance(topic_data["cons"], list):
                cons_str = " • ".join(topic_data["cons"])
                combined_conversations.append(f"User: What are the cons of {topic_name}? Bot: Some disadvantages are: {cons_str}{tokenizer.eos_token}")

            if "resources" in topic_data and isinstance(topic_data["resources"], dict):
                resources_parts = []
                for category, items in topic_data["resources"].items():
                    if isinstance(items, list):
                        resources_parts.append(f"{category.title()}: {', '.join(items)}")
                if resources_parts:
                    combined_conversations.append(f"User: What resources are there for {topic_name}? Bot: You can check out: {'; '.join(resources_parts)}{tokenizer.eos_token}")

            if "steps" in topic_data and isinstance(topic_data["steps"], list):
                steps_str = " -> ".join(topic_data["steps"]) # Use a clear separator
                combined_conversations.append(f"User: How can I start with {topic_name}? Bot: Here are the steps: {steps_str}{tokenizer.eos_token}")

            for alias in topic_data.get("aliases", []):
                if "overview" in topic_data:
                    combined_conversations.append(f"User: What is {alias}? Bot: {topic_data['overview']}{tokenizer.eos_token}")

    except Exception as e:
        print(f"Warning: Could not load or parse '{file_path}'. Error: {e}")

print(f"Loaded {len(combined_conversations)} conversations from existing knowledge base JSONs.")


# B. Load conversations from the new CSV file
print(f"Loading data from new CSV file: {NEW_CSV_PATH}")
try:
    df_new = pd.read_csv(NEW_CSV_PATH) #
    # CONFIRMED: Use 'Question' and 'Answer' columns based on your provided CSV content.
    for index, row in df_new.iterrows():
        if 'Question' in row and 'Answer' in row: # Confirmed column names for datascienceQnA.csv
            combined_conversations.append(f"User: {row['Question']} Bot: {row['Answer']}{tokenizer.eos_token}")
        else:
            print(f"Warning: Missing 'Question' or 'Answer' in row {index} of {NEW_CSV_PATH}. Skipping.")
except FileNotFoundError:
    print(f"Error: The new CSV file '{NEW_CSV_PATH}' was not found. Please ensure it exists and the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred loading or processing '{NEW_CSV_PATH}': {e}")
    exit()

print(f"Total conversations after combining: {len(combined_conversations)}")
if not combined_conversations:
    print("No training data collected. Please check your JSONs and CSV. Exiting.")
    exit()

# Create a Hugging Face Dataset
train_dataset = Dataset.from_dict({'text': combined_conversations})
print("Hugging Face Dataset created successfully.")
if len(train_dataset) > 0:
    print(f"First example of formatted data:\n{train_dataset[0]['text']}")


# --- 3. Tokenize Data ---
print("\n--- Tokenizing Data ---")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Running tokenizer on dataset"
)
print(f"Tokenization complete. Number of tokenized examples: {len(tokenized_datasets)}")

# --- 4. Initialize Data Collator ---
print("\n--- Initializing Data Collator ---")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
print("Data collator initialized.")

# --- 5. Initialize Model and Trainer ---
print("\n--- Initializing Model and Trainer ---")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded and moved to: {device}")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=False,
    report_to="none",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

print("Splitting dataset into training and validation sets (90/10 split)...")
split_datasets = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset_split = split_datasets['train']
eval_dataset_split = split_datasets['test']
print(f"Train dataset size: {len(train_dataset_split)}")
print(f"Eval dataset size: {len(eval_dataset_split)}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_split,
    eval_dataset=eval_dataset_split,
    data_collator=data_collator,
)
print("Trainer initialized.")

# --- 6. Train the Model ---
print("\n--- Starting Model Training ---")
trainer.train()
print("Model training complete.")

# --- 7. Save the Fine-tuned Model and Tokenizer ---
print(f"\n--- Saving Fine-tuned Model to {FINE_TUNED_MODEL_OUTPUT_DIR} ---")
trainer.save_model(FINE_TUNED_MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(FINE_TUNED_MODEL_OUTPUT_DIR)
print("Fine-tuned model and tokenizer saved successfully.")

print("\nFine-tuning script finished.")