import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")

ELEVENLABS_APY_KEY = config["ELEVENLABS_APY_KEY"]

def tanslation(audio_file):
    # pasar audio a texto
    transciption =[]
    try:
        model = whisper.load_model("base")
        result = model.transcibe(audio_file, fp16=False)
        tanscription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un eror transcribiendo el texto: {str(e)}")
    
    print(f"Texto original: {transcription}")
     
    # traducir texto  
    try:
        en_transcription = Translator(
            from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un eror traduciendo el texto: {str(e)}")
    
    print(f"Texto traducido a Inglés: {en_transcription}")
    
    #generar audio respuesta
    
    #usamos elevenlabs
    try:
        clilent = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = clilent.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",  # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                ),
            )
    
    except Exception as e:
        raise gr.Error (
            f"Se ha producido un eror generando el audio: {str(e)}")
    
    save_file_path = "audios/en.mp3"
    
    with open(save_file_path, "wb") as file:

        for chunk in response:
            if chunk:
                file.write(chunk)
                
    
    return save_file_path 
    
    
web = gr.Interface(
    fn=tanslation,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"),
    outputs=[],
    title="Taductor de voz",
    description="traductor de voz con IA a vaios idiomas"
)

web.launch()    
