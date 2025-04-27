import os
import logging
import groq
import tempfile
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groq_stt")

class GroqSTTProcessor:
    """Speech-to-text processor using Groq LLM for medical transcription enhancement."""
    
    def __init__(self, model_size="base"):
        # Get API key from environment
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in environment variables. Groq enhancement will not work.")
            self.client = None
        else:
            logger.info("✅ Groq API key found. GroqSTTProcessor initialized.")
            # Initialize Groq client with just the API key, no proxies or other arguments
            self.client = groq.Client(api_key=self.api_key)
            self.model = "llama-3.3-70b-versatile"
            
        # Initialize Whisper for first-stage transcription
        logger.info("Initializing Whisper model...")
        self.whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model initialized successfully")
    
    def transcribe(self, audio_path):
        """Two-stage transcription: Whisper first, then Groq enhancement."""
        try:
            # First stage: Transcribe with Whisper
            logger.info(f"Stage 1: Transcribing with Whisper: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            whisper_text = result["text"]
            logger.info(f"Whisper transcription: {whisper_text[:100]}...")
            
            # If Groq client is not available, return just the Whisper result
            if not self.client:
                logger.warning("Groq client not available, returning Whisper result only")
                return whisper_text
            
            # Second stage: Enhance with Groq
            logger.info("Stage 2: Enhancing transcription with Groq")
            return self.enhance_transcription(whisper_text)
            
        except Exception as e:
            logger.error(f"❌ Error in transcription process: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def enhance_transcription(self, text):
        """Enhance a transcription using Groq's LLM."""
        try:
            # Create a medical-focused enhancement prompt
            prompt = f"""
            I have a transcription of a patient describing endometriosis symptoms. 
            Please enhance this transcription by:
            1. Correcting any likely speech-to-text errors
            2. Identifying and properly formatting medical terms
            3. Preserving the original meaning and patient's own descriptions
            4. Organizing into a more readable format if needed
            
            Original transcription:
            "{text}"
            
            Enhanced medical transcription:
            """
            
            logger.info("Sending enhancement request to Groq")
            
            # Send to Groq for processing
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical transcription specialist with expertise in gynecology and endometriosis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for more accuracy
                max_tokens=1024
            )
            
            # Extract and return enhanced transcription
            enhanced_text = response.choices[0].message.content.strip()
            logger.info(f"Enhancement successful, length: {len(enhanced_text)} characters")
            return enhanced_text
            
        except Exception as e:
            logger.error(f"❌ Error enhancing with Groq: {str(e)}")
            # Fall back to original text on error
            logger.info("Falling back to original Whisper transcription")
            return text 