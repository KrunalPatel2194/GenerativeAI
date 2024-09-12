# train_peft_model.py
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from peft_model import create_peft_model, save_peft_model

# Load the wikitext-2-raw-v1 dataset
def load_wikitext_dataset():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Take only a small subset of the data for faster, smaller training
    train_dataset = dataset["train"].train_test_split(test_size=0.99)["train"]  # Use 1% of the train set
    eval_dataset = dataset["validation"].train_test_split(test_size=0.99)["train"]  # Use 1% of the validation set
    
    return train_dataset, eval_dataset

# Preprocess the dataset to tokenize the text
def preprocess_data(tokenizer, dataset):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_peft_model():
     # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Load the dataset
    train_dataset, eval_dataset = load_wikitext_dataset()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Preprocess the datasets
    train_dataset = preprocess_data(tokenizer, train_dataset)
    eval_dataset = preprocess_data(tokenizer, eval_dataset)
    
    # Create the PEFT model
    model = create_peft_model()
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,   # You can increase this for more epochs
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True,
    )
    
    # Define a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()

    # Save the trained model
    save_peft_model(model)

if __name__ == "__main__":
    train_peft_model()
