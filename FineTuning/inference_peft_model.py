# inference_peft_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer():
    # Load tokenizer and fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    peft_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    
    return tokenizer, peft_model

def run_inference(input_text):
    tokenizer, peft_model = load_model_and_tokenizer()
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate output
    output = peft_model.generate(**inputs)
    
    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    input_text = "Once upon a time"
    generated_text = run_inference(input_text)
    print(generated_text)
