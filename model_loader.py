from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_bloom_model(model_name="bigscience/bloom-1b1", device="cuda"):
    """
    Load a BLOOM model and its tokenizer
    
    Args:
        model_name (str): Name of the BLOOM model variant to load
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, tokenizer)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    if device == "cpu":
        model = model.to(device)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_bloom_model()
    
    text = "Once upon a time"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}") 