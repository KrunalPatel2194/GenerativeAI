# peft_config.py

from peft import LoraConfig

# Define your PEFT configuration
def create_peft_config():
    peft_config = LoraConfig(
        r=16,           # Rank of the low-rank adaptation matrices
        lora_alpha=32,  # Scaling factor for LoRA weights
        target_modules=["attn.c_attn", "mlp.c_fc"],  # Apply to specific modules
        lora_dropout=0.1,  # Dropout rate
        bias="none",   # Can be 'none', 'all', or 'lora_only'
        task_type="CAUSAL_LM"  # Task type (causal language model)
    )
    return peft_config
