import whisper
from pydub import AudioSegment
import os
import tempfile

class STTProcessor:
    def __init__(self, model_size="base"):
        print("Initializing Whisper model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model initialized successfully")

    def transcribe(self, audio_path):
        """Convert speech to text."""
        try:
            print(f"Processing audio file: {audio_path}")
            
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                wav_path = os.path.join(temp_dir, "temp.wav")
                
                try:
                    # Convert to WAV format if needed
                    if not audio_path.endswith(".wav"):
                        print("Converting audio to WAV format...")
                        audio = AudioSegment.from_file(audio_path)
                        audio = audio.set_frame_rate(16000)  # Whisper expects 16kHz
                        audio = audio.set_channels(1)        # Convert to mono
                        audio = audio.set_sample_width(2)    # Set to 16-bit
                        audio.export(wav_path, format="wav")
                        print("Audio conversion successful")
                    else:
                        wav_path = audio_path

                    # Transcribe the audio
                    print("Starting transcription...")
                    result = self.model.transcribe(wav_path)
                    print("Transcription completed successfully")
                    
                    return result["text"]
                    
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
                    raise Exception(f"Failed to process audio: {str(e)}")
                    
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise