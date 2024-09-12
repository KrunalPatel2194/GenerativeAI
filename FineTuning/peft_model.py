from transformers import AutoModelForCausalLM
from peft import get_peft_model
from peft_config import create_peft_config

def create_peft_model():
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")

    peft_config = create_peft_config()

    peft_model = get_peft_model(base_model,peft_config)

    return peft_model

def save_peft_model(model, output_dir='./fine_tuned_model'):
    model.save_pretrained(output_dir)