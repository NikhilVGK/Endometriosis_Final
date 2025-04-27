from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from groq import Groq
import os
import logging

class VisionAnalyzer:
    def __init__(self, model_path="models/combined_model.h5"):
        self.model = None
        self.client = None
        self.model_name = "llama-3.3-70b-versatile"
        
        try:
            # Initialize Groq client
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            # Load model if path exists
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logging.info("Successfully loaded vision model")
            else:
                logging.warning(f"Model file not found at {model_path}")
        except Exception as e:
            logging.error(f"Error initializing VisionAnalyzer: {str(e)}")
            # Continue without model - we can still use text extraction

    def analyze(self, image_path):
        """Analyze ultrasound/MRI image."""
        if self.model is None:
            logging.warning("Model not loaded - skipping image analysis")
            return 0.5
            
        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Placeholder for prediction
            logging.warning("VisionAnalyzer.analyze currently only prepares image input, but loaded combined model expects image AND tabular data. Prediction will likely fail.")
            return 0.5
        except Exception as e:
            logging.error(f"Error analyzing image: {str(e)}")
            return 0.5

    def extract_text_from_image(self, image_path):
        """
        Extract text from an image using Llama model with vision capabilities
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing extracted text and confidence scores
        """
        if self.client is None:
            return {
                "success": False,
                "error": "Groq client not initialized"
            }
            
        try:
            # Read the image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
            # Create a base64 encoded string of the image
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create the prompt with the image
            prompt = f"""Please analyze this image and extract any text you can find in it. 
            Describe the location and content of each text element.
            
            <image>
            {image_b64}
            </image>"""
            
            # Get completion from the model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,
                max_tokens=500
            )
            
            # Process the response
            if response.choices and response.choices[0].message.content:
                text_content = response.choices[0].message.content
                return {
                    "success": True,
                    "full_text": text_content,
                    "text_blocks": [{
                        "text": text_content,
                        "confidence": 1.0,
                        "bounds": []  # Llama doesn't provide bounding boxes
                    }]
                }
            else:
                return {
                    "success": False,
                    "error": "No text detected in the image"
                }

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing image: {str(e)}"
            }