![github-submission-banner](https://github.com/user-attachments/assets/a1493b84-e4e2-456e-a791-ce35ee2bcf2f)

# 🚀 Project Title

Endometriosis Detection Tool

## 📌 Problem Statement

Problem Statement 1- Weave AI magic with Groq

## 🎯 Objective

Endometriosis is a chronic gynecological condition that affects approximately 10% of women of reproductive age worldwide.  
Despite its prevalence, the average diagnosis delay is around 7–10 years, mainly due to:
- Non-specific symptoms like pelvic pain and infertility
- Dependence on invasive laparoscopic surgery
- Lack of integrated symptom and imaging analysis

This project aims to bridge this diagnostic gap by developing an AI-powered, non-invasive detection system using ultrasound images and patient symptom analysis to predict the likelihood of endometriosis accurately and earlier.

- Reduce diagnostic delays by providing an AI-based prediction system using ultrasound images and symptom data.
- Create a multimodal fusion model combining medical imaging and clinical symptoms to enhance diagnostic accuracy.
- Leverage Groq AI capabilities for enhanced natural language understanding, faster inference, and real-time feedback.
- Develop an accessible health tool that empowers both patients and healthcare providers.

## 🧠 Team & Approach

### Team Name:  
AI_Mavericks

### Team Members:  -
- Subhiksha Manivannan (GEN AI specialist)
- Nikhil Venkata Ganesh Konatham (Backend Developer)
- Nikita Prasad (Frontend Developer)
- Netranjali (Full Stack developer)

### Your Approach:  
1. Build an end-to-end system that accepts ultrasound scans and clinical symptom inputs.  
2. Use U-Net CNN to detect lesions from ultrasound images.  
3. Apply fine-tuned BERT models for analyzing free-text symptom descriptions.  
4. Fuse image and symptom embeddings through an attention-based multimodal model.  
5. Integrate Groq's Llama 3.3 70B model to personalize reports, enhance NLP tasks, and accelerate inference.
6. Deploy a Flask backend with a simple, responsive front-end for user interaction.
7. Ensure secure data management, patient privacy, and clinical explainability.


## 🛠️ Tech Stack

- Frontend: HTML, CSS and JAVASCRIPT
- Backend: Python (Flask)
- Database: SQLite with SQLAlchemy ORM
- AI/ML:  
  - U-Net CNN (Image Segmentation)  
  - BERT (Symptom Analysis)  
  - Fusion Model (Multimodal Data Integration)  
- Deployment:  
  - Flask REST APIs  
  - Render (optional) / Localhost
- Security:  
  - Password hashing  
  - Secure file uploads  
  - Session management

### Sponsor Technologies Used (if any):
- Groq:  
  - Llama 3.3 70B integration for:  
    - Personalized health report generation  ✅
    - Speech-to-Text medical transcription  ✅
    - Phase-specific motivational content  ✅
    - Enhancing diagnostic insights and explainability  ✅
- OpenAI (optional): For fallback NLP if Groq is unavailable.✅
- Other Libraries: TensorFlow, HuggingFace Transformers, SQLAlchemy, Flask-Login.✅


# ✨ Key Features

Highlight the most important features of your project:

- ✅ *Multimodal Diagnosis:* Combines ultrasound image analysis , text description and symptom interpretation for accurate endometriosis detection.  
- ✅ *Groq-Powered Enhancements:* AI-driven personalized reports, phase-specific emotional support tips, and fast speech-to-text processing.  
- ✅ *Comprehensive Health Management:* Pain tracking, menstrual cycle phase prediction, medication management, and prescription analysis.  
- ✅ *Community Support & Feedback:* Anonymous story sharing, user feedback system, and personalized tips for each menstrual phase.

Add images, GIFs, or screenshots if helpful!

---

## 📽️ Demo & Deliverables

- **Demo Video Link:         https://drive.google.com/file/d/1d4eupcB9UbC_ZPriq2m4S5vS7rE7uqEe/view?usp=drivesdk
- **Pitch Deck / PPT Link:   https://docs.google.com/presentation/d/1TgkTaFjSth-Ie5tEOg2B0ltAB3tnhcWV/edit?usp=drivesdk&ouid=102487277286343502717&rtpof=true&sd=true 

---

## ✅ Tasks & Bonus Checklist

- [✅ ] **All members of the team completed the mandatory task - Followed at least 2 of our social channels and filled the form** (Details in Participant Manual)  
- [✅ ] **All members of the team completed Bonus Task 1 - Sharing of Badges and filled the form (2 points)**  (Details in Participant Manual)
- [✅ ] **All members of the team completed Bonus Task 2 - Signing up for Sprint.dev and filled the form (3 points)**  (Details in Participant Manual)

*(Mark with ✅ if completed)*

---

## 🧪 How to Run the Project

### Requirements:
- the requirements.txt file is attatched, Isstall all the reuirements
- Node.js / Python / Flask
- API Keys (groq api key, EvenLabsApi Key[original key not being shared for security reasons])
- .env file setup(Groq_API_Key= 'your api key' , EVENLABS_API_KEY = ' YOur evenlabs Api key'
- Once done, Run command: python app.py , or simply run the app.py file

### Local Setup:
```bash
# Clone the repo
git clone https://github.com/your-team/project-name

# Install dependencies
cd project-name
npm install

# Start development server
npm run dev
```

Provide any backend/frontend split or environment setup notes here.

- Download all the requirements in the venv and in the the global system of the system
---

## 🧬 Future Scope

List improvements, extensions, or follow-up features:

- 📈 *Clinical Deployment:* Test EndoMetrics with real patient data in hospitals to validate practical use.
- 🛡 *Enhanced Explainability:* Integrate SHAP and Grad-CAM for better model transparency and trust.
- 🌐 *Broader Accessibility:* Add multilingual support and mobile-first responsive design for global reach.
- 🧬 *Genetic & Hormonal Data Integration:* Incorporate additional biomarkers for higher diagnostic precision.
- 🤖 *Groq-NLP Powered Chatbot:* Build an AI assistant for real-time patient support and symptom tracking.
- AI Prescription Reading -  Image is uploaded to the ai, it will scan and give the medicine in the prescription



## 📎 Resources / Credits

- *APIs/AI Models:*  
  - Groq API (Llama 3.3 70B) for enhanced NLP and fast inference  
  - Whisper API for initial speech-to-text processing  

- *Datasets:*  
  - Sample ultrasound datasets for training CNNs  
  - Symptom datasets for fine-tuning BERT NLP model  

- *Open Source Libraries/Tools:*  
  - Flask, SQLAlchemy, SQLite  
  - TensorFlow, HuggingFace Transformers, OpenCV  
  - HTML, CSS, JavaScript for frontend UI

- *Acknowledgements:*  
  - Groq and Hackhazards organizers for providing a platform to innovate.  
  - Medical literature on endometriosis for guiding symptom analysis logic.



## 🏁 Final Words


Participating in Hackhazards has been a thrilling and transformative journey.  
Building EndoMetrics challenged us to merge medical understanding with cutting-edge AI, while also learning to integrate Groq’s ultra-fast models for real-world impact.  
We faced technical hurdles like multimodal fusion and API optimizations, but overcoming them taught us resilience, teamwork, and creativity.  
We’re proud to contribute toward faster, non-invasive diagnosis for women suffering silently — and we’re excited to continue building solutions that matter! 🚀
