"""
Llama 3.3 70B Model from Groq for enhancing prediction results.
This module provides integration with Groq's API to leverage the power of Llama 3.3 70B
for more accurate endometriosis assessment and personalized insights.
"""

import os
import json
import logging
import requests
import traceback
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_enhancer')

class LlamaEnhancer:
    """
    A class to enhance prediction results using Groq's Llama 3.3 70B model.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LlamaEnhancer with Groq API credentials.
        
        Args:
            api_key: Groq API key. If None, will attempt to read from GROQ_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.error("⚠️ No Groq API key provided. LlamaEnhancer will not function.")
        else:
            logger.info("✅ Groq API key found. LlamaEnhancer initialized successfully.")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"  # Using the 70B model from Groq
        
        # Medical knowledge base for endometriosis
        self.medical_context = {
            "stages": {
                "stage_i": "Minimal: Small, superficial endometriosis implants on organs or tissue",
                "stage_ii": "Mild: Deeper implants, may involve more organs",
                "stage_iii": "Moderate: Many deep implants, small endometriomas, filmy adhesions",
                "stage_iv": "Severe: Many deep implants, large endometriomas, dense adhesions"
            },
            "symptoms": {
                "menstrual": ["Heavy bleeding", "Irregular periods", "Spotting between periods", "Blood clots"],
                "digestive": ["Bloating", "Nausea", "Constipation", "Diarrhea"],
                "urinary": ["Frequent urination", "Pain during urination", "Blood in urine"],
                "mental": ["Anxiety", "Depression", "Mood swings", "Fatigue/Low energy"],
                "sexual": ["Pain during intercourse", "Bleeding after intercourse", "Loss of libido"],
                "fertility": ["Difficulty conceiving", "Miscarriage history", "IVF treatment"]
            }
        }
    
    def _prepare_patient_context(self, patient_data: Dict[str, Any]) -> str:
        """
        Prepare a detailed context string about the patient from their data.
        
        Args:
            patient_data: Dictionary containing patient information and symptoms.
            
        Returns:
            A formatted string with patient context for the LLM.
        """
        # Extract basic information
        age = patient_data.get('age', 'unknown')
        pain_level = patient_data.get('pain_level', 'unknown')
        bmi = patient_data.get('bmi', 'unknown')
        
        # Extract reported symptoms
        symptoms = []
        for category in self.medical_context['symptoms']:
            for symptom in self.medical_context['symptoms'][category]:
                symptom_key = symptom.lower().replace(' ', '_').replace('/', '_')
                if patient_data.get(symptom_key) == 'on':
                    symptoms.append(symptom)
        
        # Extract other markers
        menstrual_irregularity = 'Yes' if patient_data.get('menstrual_irregularity') == 'on' else 'No'
        hormone_abnormality = 'Yes' if patient_data.get('hormone_abnormality') == 'on' else 'No'
        infertility = 'Yes' if patient_data.get('infertility') == 'on' else 'No'
        
        # Get text description if available
        text_description = patient_data.get('description', '')
        
        # Format context
        context = f"""
Patient Profile:
- Age: {age}
- Pain Level (0-10): {pain_level}
- BMI: {bmi}
- Menstrual Irregularity: {menstrual_irregularity}
- Hormone Abnormality: {hormone_abnormality}
- Infertility Issues: {infertility}

Reported Symptoms:
{', '.join(symptoms) if symptoms else 'None specifically reported'}

Patient's Description:
{text_description if text_description else 'No additional description provided'}
        """
        
        return context.strip()

    def enhance_prediction(self, 
                          patient_data: Dict[str, Any], 
                          ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the machine learning prediction with LLM insights.
        
        Args:
            patient_data: Dictionary containing patient information and symptoms
            ml_prediction: Dictionary containing the ML model prediction results
            
        Returns:
            Enhanced prediction with personalized insights
        """
        if not self.api_key:
            logger.error("Cannot enhance prediction: No Groq API key provided")
            return ml_prediction
        
        try:
            logger.info("Starting LLM enhancement process with Groq Llama 3.3")
            
            # Prepare context
            patient_context = self._prepare_patient_context(patient_data)
            prediction_stage = ml_prediction.get('prediction', {}).get('stage', 'Unknown')
            confidence = ml_prediction.get('prediction', {}).get('confidence', '0%')
            
            logger.info(f"Processing prediction: {prediction_stage} with confidence {confidence}")
            
            # Construct prompt for the LLM
            prompt = f"""
You are an AI medical assistant specializing in endometriosis, working with a hybrid system that combines machine learning predictions with your advanced reasoning capabilities. 

IMPORTANT FORMATTING INSTRUCTIONS:
1. DO NOT use any asterisks (*) anywhere in your response
2. DO NOT use bullet points with dashes (-) or bullets (•)
3. Use numbered points instead of bullet points
4. Use clear paragraph formatting with line breaks

The machine learning model has analyzed the patient's data and produced the following prediction:
- Stage Assessment: {prediction_stage}
- Confidence Level: {confidence}

Patient Information:
{patient_context}

Based on the ML prediction and the patient information above:

1. Provide a detailed analysis of the assessment, explaining what this stage means in terms of endometriosis progression
2. Highlight key symptoms that support this diagnosis, and note any unusual or contradictory indicators
3. Suggest 3-5 personalized recommendations for the patient
4. Include key points the patient should discuss with their healthcare provider

Respond in a compassionate, clear, and educational manner. Include both medical context and practical advice for managing symptoms.
"""
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a specialized medical AI assistant focusing on endometriosis assessment and patient education. NEVER use asterisks or bullet points in your responses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            logger.info(f"Sending request to Groq API with model: {self.model}")
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info("✅ Successfully received response from Groq API")
                result = response.json()
                llm_analysis = result['choices'][0]['message']['content']
                
                # Enhance the original prediction
                enhanced_prediction = ml_prediction.copy()
                enhanced_prediction['educational_information'] = llm_analysis
                enhanced_prediction['llm_enhanced'] = True
                enhanced_prediction['model_used'] = "Llama 3.3 70B by Groq"
                
                return enhanced_prediction
            else:
                logger.error(f"❌ Error calling Groq API: {response.status_code} - Response: {response.text}")
                # Try to parse the error for more details
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'Unknown error')
                    logger.error(f"Groq API error details: {error_message}")
                except:
                    logger.error("Could not parse error response from Groq API")
                
                # Add error information to the prediction
                ml_prediction['llm_enhanced'] = False
                ml_prediction['llm_error'] = f"API Error: {response.status_code}"
                return ml_prediction
                
        except Exception as e:
            logger.error(f"❌ Error enhancing prediction: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Add error information to the prediction
            ml_prediction['llm_enhanced'] = False
            ml_prediction['llm_error'] = f"Exception: {str(e)}"
            return ml_prediction
    
    def generate_personalized_report(self, 
                                    patient_data: Dict[str, Any], 
                                    prediction: Dict[str, Any]) -> str:
        """
        Generate a personalized report content using the Llama model.
        
        Args:
            patient_data: Dictionary containing patient information
            prediction: Dictionary containing prediction results
            
        Returns:
            Structured report content as a string
        """
        if not self.api_key:
            logger.error("Cannot generate report: No Groq API key provided")
            return "Error: Cannot generate report without Groq API key"
        
        try:
            logger.info("Starting personalized report generation with Groq Llama 3.3")
            
            # Prepare context
            patient_context = self._prepare_patient_context(patient_data)
            prediction_stage = prediction.get('prediction', {}).get('stage', 'Unknown')
            confidence = prediction.get('prediction', {}).get('confidence', '0%')
            
            # Construct prompt for the LLM
            prompt = f"""
Create a professional medical assessment report for endometriosis. Format it like a formal hospital report with clear sections.

IMPORTANT FORMATTING INSTRUCTIONS:
1. DO NOT use any asterisks (*) in the entire report
2. DO NOT use bullet points with dashes (-) or bullets (•)
3. Number all points instead of using bullets
4. Use clear paragraph formatting with line breaks

Machine Learning Assessment:
- Stage Assessment: {prediction_stage}
- Confidence Level: {confidence}

Patient Information:
{patient_context}

Format the report with these sections:

1. ASSESSMENT SUMMARY
   A concise overview of findings (1-2 paragraphs)

2. CLINICAL FINDINGS
   Key observations supporting the assessment, formatted as numbered points (not bullets or asterisks)
   
3. DETAILED ANALYSIS
   Explain what this stage means for endometriosis progression
   
4. SYMPTOMS ANALYSIS
   Identify relevant symptoms and their significance, formatted as numbered points (not bullets or asterisks)
   
5. RECOMMENDATIONS
   3-5 personalized recommendations for management, formatted as numbered points (not bullets or asterisks)
   
6. DISCUSSION POINTS
   Important topics for the patient to discuss with their healthcare provider, formatted as numbered points (not bullets or asterisks)

End with a disclaimer that this assessment is based on self-reported symptoms and algorithmic analysis, and is not a substitute for professional medical diagnosis.

Format the report in a clean, professional style that resembles a hospital electronic medical record. Use clear headings, proper spacing, and formal medical language.
"""
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a specialized medical AI assistant creating formal hospital-style assessment reports for endometriosis patients. NEVER use asterisks or bullet points in your reports."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1800
            }
            
            logger.info(f"Sending request to Groq API with model: {self.model}")
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info("✅ Successfully received response from Groq API")
                result = response.json()
                report_content = result['choices'][0]['message']['content']
                
                # Add report header and format
                formatted_report = f"""
=============================================================
ENDOMETRICS ASSESSMENT REPORT
=============================================================

{report_content}

=============================================================
Report generated by Endometrics Health Assessment System
Enhanced by Llama 3.3 70B
This report is for informational purposes only
=============================================================
"""
                return formatted_report
            else:
                logger.error(f"❌ Error calling Groq API: {response.status_code} - Response: {response.text}")
                error_message = "Unable to generate personalized report due to API error."
                return error_message
                
        except Exception as e:
            logger.error(f"❌ Error generating report: {str(e)}")
            logger.error(traceback.format_exc())
            error_message = "Unable to generate personalized report due to an unexpected error."
            return error_message 

    def get_response(self, prompt, temperature=0.7, max_tokens=350):
        """
        Get a simple text response from the Groq API.
        
        Args:
            prompt: The text prompt to send to the model
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text response
        """
        if not self.api_key:
            logger.error("Cannot get response: No Groq API key provided")
            return "Error: No Groq API key available"
        
        try:
            logger.info(f"Sending prompt to Groq (temp={temperature}, max_tokens={max_tokens})")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specializing in women's health."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                logger.info("Successfully received response from Groq API")
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Error calling Groq API: {response.status_code} - Response: {response.text}")
                return f"Error: API returned status code {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}" 