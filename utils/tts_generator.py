from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()

class TTSGenerator:
    def __init__(self):
        self.client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )

    def generate_audio(self, text, voice="Rachel", model="eleven_multilingual_v2", save_path=None):
        """Convert text to speech using the client instance."""
        audio = self.client.generate(
            text=text,
            voice=voice,
            model=model
        )
        if save_path:
            with open(save_path, "wb") as f:
                f.write(audio)
        return audio