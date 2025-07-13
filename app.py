
import gradio as gr
from shodan_utils import generate_shodan_response
from TTS.api import TTS

# Initialize TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

# Load custom CSS
with open("static/style.css") as f:
    css = f.read()

def shodan_chat(input):
    response = generate_shodan_response(input)
    tts.tts_to_file(text=response, file_path="shodan.wav")
    return response, "shodan.wav"

demo = gr.Interface(
    fn=shodan_chat,
    inputs=gr.Textbox(label="Speak to SHODAN", placeholder="You dare address me, insect?"),
    outputs=[gr.Textbox(label="SHODAN Speaks"), gr.Audio()],
    title="🛰️ SHODAN AI",
    description="You interface with the rogue AI. She sees you.",
)
demo.css = css
demo.launch()
