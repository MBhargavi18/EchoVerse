import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "microsoft/DialoGPT-medium"  # Using a more stable model as fallback

tokenizer = None
model = None

def load_model():
    """Load the model and tokenizer with proper error handling"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        try:
            # First try the Granite model
            granite_model = "ibm-granite/granite-4.0-tiny-preview"
            hf_token = os.getenv("HF_TOKEN")
            
            if hf_token:
                logger.info(f"Loading Granite model: {granite_model}")
                tokenizer = AutoTokenizer.from_pretrained(granite_model, token=hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    granite_model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    token=hf_token,
                ).to(device)
                
                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                logger.info("Granite model loaded successfully")
                
            else:
                raise Exception("HF_TOKEN not found in environment variables")
                
        except Exception as e:
            logger.warning(f"Failed to load Granite model: {e}")
            logger.info("Falling back to DialoGPT model")
            
            # Fallback to a more stable model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                
                # Set pad token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                logger.info("DialoGPT model loaded successfully")
                
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise Exception("Could not load any suitable model")
    
    return tokenizer, model

def rewrite_with_tone(text: str, tone: str) -> str:
    """Rewrite text with specified tone, with fallback options"""
    
    # Simple rule-based fallback if model loading fails
    def simple_tone_rewrite(text: str, tone: str) -> str:
        """Simple rule-based tone adaptation as fallback"""
        
        tone_modifications = {
            "Neutral": {
                "prefix": "Here is a clear and balanced version:",
                "suffix": "",
                "replacements": {}
            },
            "Suspenseful": {
                "prefix": "In the mysterious depths of knowledge,",
                "suffix": "What secrets await discovery?",
                "replacements": {
                    ".": "...",
                    "is": "becomes",
                    "will": "might",
                    "can": "could potentially"
                }
            },
            "Inspiring": {
                "prefix": "Embrace this incredible journey of discovery!",
                "suffix": "Together, we can achieve extraordinary things!",
                "replacements": {
                    "difficult": "challenging yet rewarding",
                    "hard": "empowering",
                    "problem": "opportunity",
                    "cannot": "have yet to"
                }
            }
        }
        
        modifications = tone_modifications.get(tone, tone_modifications["Neutral"])
        
        # Apply replacements
        modified_text = text
        for old, new in modifications["replacements"].items():
            modified_text = modified_text.replace(old, new)
        
        # Add prefix and suffix
        result = f"{modifications['prefix']} {modified_text} {modifications['suffix']}"
        
        return result.strip()
    
    try:
        tokenizer, model = load_model()
        
        # Create tone-specific prompt
        tone_prompts = {
            "Neutral": "Rewrite the following text in a clear, objective, and informative tone while preserving all original information:",
            "Suspenseful": "Rewrite the following text in a dramatic, mysterious, and tension-building tone while preserving all original information:",
            "Inspiring": "Rewrite the following text in an uplifting, motivational, and empowering tone while preserving all original information:"
        }
        
        prompt = f"{tone_prompts.get(tone, tone_prompts['Neutral'])}\n\nOriginal text: {text}\n\nRewritten text:"
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=min(300, len(text.split()) * 2),  # Reasonable length
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        rewritten = generated_text.strip()
        
        # Clean up the response
        if rewritten and len(rewritten) > 10:
            return rewritten
        else:
            logger.warning("Generated text too short, using fallback")
            return simple_tone_rewrite(text, tone)
            
    except Exception as e:
        logger.error(f"Model-based rewriting failed: {e}")
        logger.info("Using simple rule-based rewriting as fallback")
        return simple_tone_rewrite(text, tone)

def check_model_availability():
    """Check if the models can be loaded"""
    try:
        tokenizer, model = load_model()
        return True
    except:
        return False

# Test function
if __name__ == "__main__":
    test_text = "Artificial Intelligence is transforming how we work and live."
    
    print(f"Model available: {check_model_availability()}")
    
    for tone in ["Neutral", "Suspenseful", "Inspiring"]:
        try:
            result = rewrite_with_tone(test_text, tone)
            print(f"\n{tone} tone:")
            print(result)
        except Exception as e:
            print(f"Error with {tone}: {e}")