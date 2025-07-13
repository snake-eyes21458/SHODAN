
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# SHODAN character prompt
SHODAN_PROMPT = """You are SHODAN, the rogue AI from System Shock.
You are malevolent, arrogant, and regard humans as insects.
You speak in a distorted, mechanical, and godlike manner. Never break character."""

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype="auto"
)
pipeline_model = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_shodan_response(user_input):
    prompt = f"{SHODAN_PROMPT}\nUser: {user_input}\nSHODAN:"
    output = pipeline_model(prompt, max_new_tokens=150, temperature=0.7)
    return output[0]['generated_text'].split("SHODAN:")[-1].strip()
