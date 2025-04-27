import os
import json
import requests
from typing import Dict, Any, Optional

class GroqPrescriptionAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq Prescription Analyzer.
        
        Args:
            api_key (str, optional): Groq API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it as GROQ_API_KEY environment variable or pass it to the constructor.")
        
        self.base_url = "https://api.groq.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_prescription(self, prescription_text: str) -> Dict[str, Any]:
        """
        Analyze a prescription using Groq's API.
        
        Args:
            prescription_text (str): The text content of the prescription to analyze.
            
        Returns:
            Dict[str, Any]: Analysis results including medications, dosages, and instructions.
        """
        try:
            # Prepare the prompt for the API
            prompt = f"""
            Analyze the following prescription and extract the following information:
            1. List of medications with their dosages
            2. Frequency of administration
            3. Special instructions
            4. Duration of treatment
            
            Prescription:
            {prescription_text}
            
            Please provide the analysis in JSON format.
            """
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "You are a medical prescription analyzer. Extract and structure prescription information accurately."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Try to parse the JSON response
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, return it as text
                analysis = {
                    "raw_analysis": analysis_text,
                    "error": "Could not parse response as JSON"
                }
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}"
            }

    def validate_prescription(self, prescription_text: str) -> Dict[str, Any]:
        """
        Validate a prescription for potential issues or errors.
        
        Args:
            prescription_text (str): The text content of the prescription to validate.
            
        Returns:
            Dict[str, Any]: Validation results including any warnings or errors found.
        """
        try:
            # Prepare the prompt for validation
            prompt = f"""
            Analyze the following prescription for potential issues or errors:
            1. Check for missing or unclear dosages
            2. Verify frequency of administration is specified
            3. Look for potential drug interactions
            4. Check for any contraindications
            5. Verify duration of treatment is specified
            
            Prescription:
            {prescription_text}
            
            Please provide the validation results in JSON format.
            """
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "You are a medical prescription validator. Identify potential issues in prescriptions."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            validation_text = result['choices'][0]['message']['content']
            
            # Try to parse the JSON response
            try:
                validation = json.loads(validation_text)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, return it as text
                validation = {
                    "raw_validation": validation_text,
                    "error": "Could not parse response as JSON"
                }
            
            return {
                "success": True,
                "validation": validation
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}"
            } 