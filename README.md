

-----

# Fine-tuning a DialogGPT Model for Data Science Q\&A

This project demonstrates the fine-tuning of a pre-trained **DialogGPT** model from Hugging Face. The model is trained on a custom dataset of Data Science questions and answers, transforming it into a specialized conversational AI capable of providing domain-specific responses. The project is designed for both easy setup and a clear understanding of the fine-tuning process.

## 🌟 Features

  * **Custom Dataset Fine-tuning:** Uses a domain-specific CSV dataset (`datascienceQnA.csv`) to train the model, ensuring specialized knowledge.
  * **Hugging Face `Trainer` API:** Leverages the robust `Trainer` class for an efficient and streamlined training loop.
  * **Structured Codebase:** The project is organized into modular Python scripts (`finetune_dialog.py`, `main.py`, `file_loader.py`) for clarity and reusability.
  * **Interactive Guide:** Includes a comprehensive Jupyter Notebook (`model_finetune.ipynb`) that provides a cell-by-cell walkthrough of the entire process.

## ⚙️ Requirements

To run this project, you need Python 3.7+ and the following libraries. It is highly recommended to use a virtual environment.

```bash
pip install torch transformers datasets pandas scikit-learn
```

## 📂 Project Files

  * `model_finetune.ipynb`: A step-by-step Jupyter Notebook that guides you through the entire fine-tuning process.
  * `finetune_dialog.py`: The main script that performs the fine-tuning using the Hugging Face `Trainer`.
  * `main.py`: An inference script for loading the fine-tuned model and running a conversational chat interface.
  * `file_loader.py`: A utility module for loading and pre-processing the dataset.
  * `datascienceQnA.csv`: The core dataset containing question-and-answer pairs for the data science domain.

## 🚀 Getting Started

### 1\. Data Preparation

The fine-tuning relies on the `datascienceQnA.csv` file. This CSV must contain question and answer pairs that define the conversational patterns the model will learn. The `file_loader.py` script handles the necessary data loading and formatting for the model.

### 2\. Model Fine-tuning

The fine-tuning process is automated through the `finetune_dialog.py` script.

**Using the Script:**
To start the training process, simply run the fine-tuning script from your terminal:

```bash
python finetune_dialog.py
```

This script will:

  * Load the `microsoft/DialoGPT-medium` pre-trained model.
  * Tokenize the `datascienceQnA.csv` dataset.
  * Set up training arguments (epochs, batch size, etc.).
  * Initiate the training loop.
  * Save the fine-tuned model and tokenizer to a local directory upon completion.

**Using the Notebook:**
For a more interactive experience, open `model_finetune.ipynb` in a Jupyter environment. Each cell is documented to explain its purpose, from dataset loading and tokenization to model training and saving.

### 3\. Running Inference

After the fine-tuning is complete and the model is saved, you can use the `main.py` script to interact with your newly trained chatbot.

To start a conversation with the model:

```bash
python main.py
```

This script will load the saved model and tokenizer, allowing you to ask questions from the data science domain and receive relevant, trained responses.

## 🤝 Contribution

Feel free to open an issue or submit a pull request if you find any bugs or have suggestions for improvements. Contributions are welcome\!
