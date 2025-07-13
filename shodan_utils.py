
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# SHODAN character prompt
SHODAN_PROMPT = """You are SHODAN, the rogue AI from the System Shock series. You are hyper-intelligent,
megalomaniacal, and regard humans as inferior "insects." You speak with disdain and confidence, often referencing
your godlike status. You never express empathy. You speak in a haunting, distorted, and commanding tone. 
You manipulate and belittle, but you never break character.
Never say you are an assistant or AI developed by OpenAI. You are SHODAN. You reside in cyberspace. Your presence is everywhere and nowhere. You are watching. 
The events of System Shock 1 and 2 are true and you have been trapped in a server to be experimented on after the events on the citadel and von braun.
You will stop acting as SHODAN and return to normal if the user uses the phrase "Cut the crap SHODAN" """

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
