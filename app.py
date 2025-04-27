from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash, session, make_response, Response, send_file
from utils.tts_generator import TTSGenerator
from utils.vision_analyzer import VisionAnalyzer
from utils.llm_enhancer import LlamaEnhancer
from utils.stt_processor import STTProcessor
from utils.groq_stt_processor import GroqSTTProcessor
from utils.groq_prescription_analyzer import GroqPrescriptionAnalyzer
import os
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    TextVectorization, Input, Embedding, LSTM, Dense, concatenate,
    Conv2D, MaxPooling2D, Flatten, Dropout
)
from tensorflow.keras.models import Model
import tempfile
import json
from pathlib import Path
from flask_cors import CORS
import logging
import shutil
import base64
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, func, desc
import random
import re
import uuid
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('endometrics')

# Function to ensure required static files and folders exist
def ensure_static_files():
    """Create necessary directories and default files if they don't exist"""
    # Ensure static folder exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Ensure images folder exists
    images_dir = os.path.join(static_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Ensure avatars folder exists
    avatars_dir = os.path.join(static_dir, 'avatars')
    os.makedirs(avatars_dir, exist_ok=True)
    
    # Ensure profile_pictures folder exists
    profile_dir = os.path.join(static_dir, 'uploads', 'profile_pictures')
    os.makedirs(profile_dir, exist_ok=True)
    
    # Create default avatar if it doesn't exist
    default_avatar_path = os.path.join(images_dir, 'default_avatar.png')
    if not os.path.exists(default_avatar_path):
        # Create a simple default avatar - a colored circle with text
        img_size = 200
        img = Image.new('RGBA', (img_size, img_size), (240, 240, 240, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a circle with a light blue background
        draw.ellipse((10, 10, img_size-10, img_size-10), fill=(173, 216, 230))
        
        # Save the image
        img.save(default_avatar_path, 'PNG')
        logger.info(f"Created default avatar at {default_avatar_path}")
    
    # Create numbered avatars 1-5 if they don't exist
    for i in range(1, 6):
        avatar_path = os.path.join(avatars_dir, f'avatar_{i}.jpg')
        if not os.path.exists(avatar_path):
            # Create a simple colored circle avatar
            img_size = 200
            img = Image.new('RGB', (img_size, img_size), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Use different colors for each numbered avatar
            colors = [
                (255, 99, 71),   # Tomato
                (60, 179, 113),  # Medium Sea Green
                (106, 90, 205),  # Slate Blue
                (255, 165, 0),   # Orange
                (218, 112, 214)  # Orchid
            ]
            
            # Draw a circle with the color
            draw.ellipse((10, 10, img_size-10, img_size-10), fill=colors[i-1])
            
            # Save the image
            img.save(avatar_path, 'JPEG')
            logger.info(f"Created avatar_{i} at {avatar_path}")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Call the function to ensure static files exist
ensure_static_files()

# Constants from training (Moved back here)
IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
TABULAR_FEATURES = 6
MAX_WORDS = 1000
MAX_LENGTH = 100
EMBEDDING_DIM = 64

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'endometrics_default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///endometrics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# File upload configuration
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads', 'prescriptions')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename, allowed_extensions):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

db = SQLAlchemy(app)


# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    date_of_birth = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_picture = db.Column(db.String(255))  # Path to user-uploaded image
    avatar_choice = db.Column(db.Integer, default=0)  # Index of selected avatar if no custom image
    
    # Relationships
    assessments = db.relationship('Assessment', backref='user', lazy=True)
    medications = db.relationship('Medication', backref='user', lazy=True)
    med_taken_records = db.relationship('MedicationTaken', backref='user', lazy=True)
    
    def set_password(self, password):
        try:
            # Ensure password is a string
            if not isinstance(password, str):
                password = str(password)
            # Generate hash with consistent method and salt length
            self.password_hash = generate_password_hash(
                password,
                method='pbkdf2:sha256',
                salt_length=16
            )
            # Verify the hash immediately after setting
            if not check_password_hash(self.password_hash, password):
                raise ValueError("Generated password hash verification failed")
        except Exception as e:
            print(f"Error setting password: {str(e)}")
            raise e
        
    def check_password(self, password):
        try:
            # Ensure password is a string
            if not isinstance(password, str):
                password = str(password)
            # Verify the hash
            return check_password_hash(self.password_hash, password)
        except Exception as e:
            print(f"Error checking password: {str(e)}")
            return False
    
    def __repr__(self):
        return f'<User {self.username}>'

class MenstrualCycle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    last_period_start = db.Column(db.Date, nullable=False)
    cycle_length = db.Column(db.Integer, default=28)  # Default 28 days
    period_duration = db.Column(db.Integer, default=5)  # Default 5 days
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref='menstrual_cycles', lazy=True)
    
    def __repr__(self):
        return f'<MenstrualCycle {self.id} for User {self.user_id}>'
    
    def get_current_phase(self):
        """Calculate and return the current menstrual phase"""
        today = datetime.utcnow().date()
        days_since_period = (today - self.last_period_start).days
        
        if days_since_period < 0:
            return {'phase': 'Invalid date', 'days': 0}
        
        # Calculate the current day in the cycle (1-based)
        current_day = (days_since_period % self.cycle_length) + 1
        
        # Menstrual Phase (Days 1 to period_duration)
        if current_day <= self.period_duration:
            # For menstrual phase, days should be 1 to period_duration
            return {'phase': 'Menstrual Phase', 'days': current_day}
        
        # Follicular Phase (Days period_duration+1 to ovulation_start-1)
        ovulation_start = 14  # Typically day 14
        if current_day < ovulation_start:
            return {'phase': 'Follicular Phase', 'days': current_day - self.period_duration}
        
        # Ovulation Phase (Days ovulation_start to ovulation_start+2)
        ovulation_end = ovulation_start + 2
        if current_day <= ovulation_end:
            return {'phase': 'Ovulation Phase', 'days': current_day - ovulation_start + 1}
        
        # Luteal Phase (Days ovulation_end+1 to cycle_length)
        return {'phase': 'Luteal Phase', 'days': current_day - ovulation_end}
    
    def get_next_phase(self):
        """Calculate and return the next menstrual phase and its start date"""
        today = datetime.utcnow().date()
        days_since_period = (today - self.last_period_start).days
        
        if days_since_period < 0:
            return {'phase': 'Invalid date', 'date': None}
        
        # Calculate the current day in the cycle (1-based)
        current_day = (days_since_period % self.cycle_length) + 1
        
        # Determine current phase and calculate next phase
        if current_day <= self.period_duration:
            # Currently in Menstrual Phase
            next_phase = 'Follicular Phase'
            next_date = self.last_period_start + timedelta(days=self.period_duration)
        elif current_day < 14:
            # Currently in Follicular Phase
            next_phase = 'Ovulation Phase'
            next_date = self.last_period_start + timedelta(days=14)
        elif current_day <= 16:
            # Currently in Ovulation Phase
            next_phase = 'Luteal Phase'
            next_date = self.last_period_start + timedelta(days=17)
        else:
            # Currently in Luteal Phase
            next_phase = 'Menstrual Phase'
            next_date = self.last_period_start + timedelta(days=self.cycle_length)
        
        return {'phase': next_phase, 'date': next_date}

# Assessment model for storing user health assessments
class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    assessment_hour = db.Column(db.Integer, default=0)  # Hour of the day (0-23)
    
    # Pain details
    pain_level = db.Column(db.Integer)  # Scale 0-10
    pain_location = db.Column(db.String(100))
    pain_description = db.Column(db.Text)
    
    # Symptoms
    menstrual_irregularity = db.Column(db.Boolean, default=False)
    hormone_abnormality = db.Column(db.Boolean, default=False)
    infertility = db.Column(db.Boolean, default=False)
    
    # Additional data
    symptoms = db.Column(db.Text)  # JSON string of symptoms
    medications_taken = db.Column(db.Text)  # JSON string of medications taken
    notes = db.Column(db.Text)
    
    # Model prediction result
    prediction_result = db.Column(db.String(100))  # e.g., "Stage I (Minimal)"
    confidence_score = db.Column(db.Float)  # 0-1
    report_content = db.Column(db.Text)  # Stored generated report content
    
    # Source of assessment creation
    assessment_source = db.Column(db.String(50), default="records")  # "records" or "assessment"
    
    def __repr__(self):
        return f'<Assessment {self.id} for User {self.user_id}>'
    
    def to_dict(self):
        """Convert assessment to dictionary for API responses"""
        return {
            'id': self.id,
            'date': self.assessment_date.strftime('%Y-%m-%d'),
            'hour': self.assessment_hour,
            'pain_level': self.pain_level,
            'pain_location': self.pain_location,
            'prediction': self.prediction_result,
            'confidence': f"{self.confidence_score * 100:.1f}%" if self.confidence_score else None,
            'symptoms': json.loads(self.symptoms) if self.symptoms else {},
            'notes': self.notes
        }

# Medication model for tracking medications
class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    frequency = db.Column(db.String(50))  # e.g., "Once daily", "Twice daily"
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    active = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Medication {self.name} for User {self.user_id}>'
    
    def to_dict(self):
        """Convert medication to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'dosage': self.dosage,
            'frequency': self.frequency,
            'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else None,
            'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else None,
            'active': self.active,
            'notes': self.notes
        }

# Medication taken tracking model
class MedicationTaken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    taken_date = db.Column(db.Date, default=datetime.utcnow().date)
    times_taken = db.Column(db.Integer, default=1)  # Track number of doses taken today
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    medication = db.relationship('Medication', backref='taken_records', lazy=True)
    
    def __repr__(self):
        return f'<MedicationTaken {self.medication_id} on {self.taken_date} ({self.times_taken} times)>'

# User Story model for sharing experiences
class Story(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    allow_sharing = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='stories', lazy=True)
    
    def __repr__(self):
        return f'<Story {self.id} by User {self.user_id}>'
    
    def to_dict(self):
        """Convert story to dictionary for API responses"""
        return {
            'id': self.id,
            'content': self.content,
            'created_at': self.created_at.strftime('%Y-%m-%d'),
            'user_first_name': self.user.first_name if self.user.first_name else 'Anonymous User',
            'allow_sharing': self.allow_sharing
        }

# Feedback model for user feedback
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Nullable to allow anonymous feedback
    category = db.Column(db.String(50), nullable=False)  # UI/UX, Features, Bug Reports, etc.
    rating = db.Column(db.Integer)  # Rating scale 1-5
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='Pending')  # Pending, Reviewed, Addressed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='feedback', lazy=True)
    
    def __repr__(self):
        return f'<Feedback {self.id} - {self.subject}>'
    
    def to_dict(self):
        """Convert feedback to dictionary for API responses"""
        return {
            'id': self.id,
            'category': self.category,
            'rating': self.rating,
            'subject': self.subject,
            'message': self.message,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d'),
            'user_name': self.user.first_name if self.user and self.user.first_name else 'Anonymous User'
        }

# Testimonial model for public testimonials
class Testimonial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))  # Can be anonymous
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)  # 1-5 stars
    avatar = db.Column(db.String(255), default='default_avatar.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Testimonial {self.id} by {self.name or "Anonymous"}>'
    
    def to_dict(self):
        """Convert testimonial to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name or 'Anonymous User',
            'content': self.content,
            'rating': self.rating,
            'avatar': self.avatar,
            'created_at': self.created_at.strftime('%Y-%m-%d')
        }

# Initialize services
tts = TTSGenerator()
vision = VisionAnalyzer()
llm_enhancer = LlamaEnhancer()
stt = STTProcessor()
groq_stt = GroqSTTProcessor()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # Direct redirect to login without flash message
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def build_image_model(input_shape):
    image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_features = Dense(128, activation='relu', name='image_features')(x)
    return Model(inputs=image_input, outputs=image_features)

def build_tabular_model(input_shape):
    tabular_input = Input(shape=(input_shape,), name='tabular_input')
    x = Dense(64, activation='relu')(tabular_input)
    x = Dropout(0.5)(x)
    tabular_features = Dense(32, activation='relu', name='tabular_features')(x)
    return Model(inputs=tabular_input, outputs=tabular_features)

def build_text_model():
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    vectorize_layer = TextVectorization(
        max_tokens=MAX_WORDS,
        output_mode='int',
        output_sequence_length=MAX_LENGTH,
        standardize='lower_and_strip_punctuation'
    )
    x = vectorize_layer(text_input)
    x = Embedding(MAX_WORDS + 1, EMBEDDING_DIM)(x)
    x = LSTM(32)(x)
    text_features = Dense(32, activation='relu', name='text_features')(x)
    return Model(inputs=text_input, outputs=text_features), vectorize_layer

def build_combined_model(image_model, tabular_model, text_model):
    combined_features = concatenate([
        image_model.output,
        tabular_model.output,
        text_model.output
    ])
    x = Dense(64, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='combined_output')(x)
    return Model(
        inputs=[image_model.input, tabular_model.input, text_model.input],
        outputs=output
    )

# Initialize models
print("Building model architecture...")
image_model = build_image_model(IMG_SHAPE)
tabular_model = build_tabular_model(TABULAR_FEATURES)
text_model, vectorize_layer = build_text_model()

# Load vectorization config and adapt layer
print("Loading vectorization config...")
config_path = 'models/vectorize_config.pkl'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Vectorize config not found at {config_path}")
with open(config_path, 'rb') as f:
    vectorize_config = pickle.load(f)

# Initialize vectorization layer with saved vocabulary
print("Initializing vectorization layer...")
vectorize_layer.adapt(tf.constant(["dummy text"]))  # Initialize layer
vectorize_layer.set_vocabulary(vectorize_config['vocabulary'])

# Build combined model
print("Building combined model...")
combined_model = build_combined_model(image_model, tabular_model, text_model)

# Load trained weights
print("Loading model weights...")
weights_path = 'models/combined_model.h5'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}")
combined_model.load_weights(weights_path)

# Initialize scaler
scaler = StandardScaler()
dummy_data = np.array([[30, 5, 25]])
scaler.fit(dummy_data)

def verify_database_password(username, password):
    """Verify if a password matches the stored hash in the database"""
    try:
        # Get the user from database
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"No user found with username: {username}")
            return False
            
        # Use the User model's check_password method
        is_valid = user.check_password(password)
        if not is_valid:
            print(f"Password verification failed for user: {username}")
            print(f"Stored hash: {user.password_hash}")
            print(f"Attempted password: {password}")
        return is_valid
    except Exception as e:
        print(f"Error verifying password: {str(e)}")
        return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to index
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Attempting login for username: {username}")
        
        # Use the verification function
        if verify_database_password(username, password):
            user = User.query.filter_by(username=username).first()
            session['user_id'] = user.id
            session['username'] = user.username
            
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

def check_database_integrity():
    """Check and fix any password hash issues in the database"""
    try:
        users = User.query.all()
        print(f"Found {len(users)} users in database")
        
        for user in users:
            print(f"\nChecking user: {user.username}")
            print(f"Current hash: {user.password_hash}")
            
            # Try to verify the hash format
            try:
                # This will raise an error if the hash is invalid
                check_password_hash(user.password_hash, "dummy")
                print("Hash format is valid")
            except Exception as e:
                print(f"Invalid hash format: {str(e)}")
                # Generate a new hash
                new_hash = generate_password_hash("default_password", method='pbkdf2:sha256', salt_length=16)
                print(f"Generated new hash: {new_hash}")
                user.password_hash = new_hash
                db.session.commit()
                print("Updated hash in database")
                
    except Exception as e:
        print(f"Error checking database: {str(e)}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        print(f"Registration attempt for username: {username}")
        print(f"Password received: {password}")
        
        # Validate input
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html', 
                                 username=username if username else "",
                                 email=email if email else "", 
                                 first_name=first_name, 
                                 last_name=last_name)
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html', username=username, email=email, 
                                 first_name=first_name, last_name=last_name)
            
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
            return render_template('register.html', email=email, 
                                 first_name=first_name, last_name=last_name)
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html', username=username, 
                                 first_name=first_name, last_name=last_name)
            
        try:
            # Create new user
            new_user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name
            )
            print(f"Created new user: {username}")
            
            # Set password with error handling
            try:
                # Ensure password is a string
                if not isinstance(password, str):
                    password = str(password)
                new_user.set_password(password)
                print(f"Password set successfully for user: {username}")
                print(f"Generated hash: {new_user.password_hash}")
            except Exception as e:
                print(f"Error setting password: {str(e)}")
                flash('Error creating account. Please try again.', 'danger')
                return render_template('register.html', 
                                     username=username, email=email,
                                     first_name=first_name, last_name=last_name)
            
            db.session.add(new_user)
            db.session.commit()
            print(f"User {username} registered successfully")
            
            # Verify the password was stored correctly
            stored_user = User.query.filter_by(username=username).first()
            if stored_user and stored_user.check_password(password):
                print("Password verification successful after registration")
            else:
                print("Password verification failed after registration")
                db.session.delete(stored_user)
                db.session.commit()
                flash('Error creating account. Please try again.', 'danger')
                return render_template('register.html', 
                                     username=username, email=email,
                                     first_name=first_name, last_name=last_name)
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Error during registration: {str(e)}")
            flash('Error creating account. Please try again.', 'danger')
            return render_template('register.html', 
                                 username=username, email=email,
                                 first_name=first_name, last_name=last_name)
        
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    
    # Get user's assessments, ordered by date (most recent first)
    assessments = Assessment.query.filter_by(
        user_id=session['user_id']
    ).order_by(Assessment.assessment_date.desc()).all()
    
    # Get only assessments created from the Assessment page
    assessments = Assessment.query.filter_by(
        user_id=session['user_id'],
        assessment_source="assessment"
    ).order_by(Assessment.assessment_date.desc()).all()
    
    return render_template('profile.html', user=user, assessments=assessments)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])
    
    # Get form data
    username = request.form.get('username')
    email = request.form.get('email')
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    avatar_choice = request.form.get('avatar_choice')
    
    # Check if username is being changed and already exists
    if username != user.username and User.query.filter_by(username=username).first():
        flash('Username already taken.', 'danger')
        return redirect(url_for('profile'))
    
    # Check if email is being changed and already exists
    if email != user.email and User.query.filter_by(email=email).first():
        flash('Email already registered.', 'danger')
        return redirect(url_for('profile'))
    
    # Update user information
    user.username = username
    user.email = email
    user.first_name = first_name
    user.last_name = last_name
    
    # Handle profile picture upload
    if 'profile_picture' in request.files and request.files['profile_picture'].filename:
        picture_file = request.files['profile_picture']
        
        # Generate secure filename
        filename = secure_filename(picture_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        new_filename = f"{user.id}_{timestamp}_{filename}"
        
        # Ensure upload directory exists
        upload_folder = os.path.join(app.static_folder, 'uploads', 'profile_pictures')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save the file
        picture_path = os.path.join(upload_folder, new_filename)
        picture_file.save(picture_path)
        
        # Update database with relative path
        user.profile_picture = f'uploads/profile_pictures/{new_filename}'
        user.avatar_choice = 0  # Reset avatar choice when uploading custom image
    
    # Handle avatar selection
    elif avatar_choice and int(avatar_choice) > 0:
        user.avatar_choice = int(avatar_choice)
        user.profile_picture = None  # Clear custom profile picture when selecting avatar
    
    # Save changes to database
    db.session.commit()
    
    # Update session username if changed
    session['username'] = user.username
    
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

# Protected routes
@app.route('/')
def index():
    # Get user ID from session if logged in
    user_id = session.get('user_id')
    
    # If user is not logged in, redirect to login page
    if not user_id:
        return redirect(url_for('login'))
    
    cycle_info = None
    med_reminders = []
    testimonials = []
    username = session.get('username', 'Guest')
    
    # Fetch user details
    user = User.query.get(user_id)
    if user:
        username = user.username
        
        # Fetch menstrual cycle information
        cycle_info = get_cycle_info(user_id)
        
        # Fetch medication reminders for today
        today_date = datetime.utcnow().date()
        active_meds = Medication.query.filter_by(user_id=user_id, active=True).all()
        
        # Check which active medications are scheduled for today based on frequency
        med_reminders = []
        for med in active_meds:
            is_due_today = True # Assuming daily for now, implement logic if needed
            
            if is_due_today:
                # Check if already taken today
                taken_record = MedicationTaken.query.filter_by(
                    user_id=user_id,
                    medication_id=med.id,
                    taken_date=today_date
                ).first()
                
                # Determine required times based on frequency
                required_times = get_required_times(med.frequency)
                times_taken = taken_record.times_taken if taken_record else 0
                
                med_reminders.append({
                    'id': med.id,
                    'name': med.name,
                    'dosage': med.dosage,
                    'frequency': med.frequency,
                    'taken_today': taken_record is not None,
                    'required_times': required_times,
                    'times_taken': times_taken,
                    'remaining': required_times - times_taken
                })

    # Fetch shared stories and process for template
    shared_stories_data = []
    shared_stories = Story.query.filter_by(allow_sharing=True).order_by(Story.created_at.desc()).limit(10).all()
    for story in shared_stories:
        user = User.query.get(story.user_id)
        user_first_name = "Anonymous"
        avatar = 'images/default_avatar.png' # Default

        if user:
            user_first_name = user.first_name if user.first_name else "Anonymous"
            logger.info(f"[Story ID: {story.id}] Checking User ID {user.id} ({user.username})")
            # Determine avatar path carefully
            if user.profile_picture:
                avatar = user.profile_picture
                logger.info(f"---> Using profile_picture: '{avatar}'")
            elif user.avatar_choice > 0:
                avatar = f'avatars/avatar_{user.avatar_choice}.jpg'
                logger.info(f"---> Using avatar_choice {user.avatar_choice}: '{avatar}'")
            else:
                avatar = 'images/default_avatar.png' # Explicit default
                logger.info(f"---> Using default avatar: '{avatar}'")
        else:
            # Log missing user for story
            logger.warning(f"[Story ID: {story.id}] User ID {story.user_id} not found. Using default avatar.")

        created_at_formatted = story.created_at.strftime('%Y-%m-%d')
        shared_stories_data.append({
            'content': story.content,
            'user_first_name': user_first_name,
            'avatar': avatar,
            'created_at': created_at_formatted
        })
        
    # Fetch testimonials -> Changed to fetch only highly-rated Feedback
    testimonials_data = [] # Keep variable name for template compatibility
    
    # Fetch from Feedback model (rating >= 4, limit 6)
    recent_feedback = Feedback.query.filter(
        Feedback.rating >= 4
    ).order_by(Feedback.created_at.desc()).limit(6).all()

    for feedback in recent_feedback:
        user_name = "Anonymous"
        avatar_path = 'images/default_avatar.png' # Default avatar
        
        # Get user details if feedback is not anonymous
        if feedback.user_id:
            user = User.query.get(feedback.user_id)
            if user:
                user_name = user.first_name if user.first_name else "Anonymous"
                logger.info(f"[Feedback ID: {feedback.id}] Checking User ID {user.id} ({user.username})")
                # Determine avatar path
                if user.profile_picture:
                    avatar_path = user.profile_picture
                    logger.info(f"---> Using profile_picture: '{avatar_path}'")
                elif user.avatar_choice > 0:
                    avatar_path = f'avatars/avatar_{user.avatar_choice}.jpg'
                    logger.info(f"---> Using avatar_choice {user.avatar_choice}: '{avatar_path}'")
                else:
                    avatar_path = 'images/default_avatar.png' # Explicit default
                    logger.info(f"---> Using default avatar: '{avatar_path}'")
            else:
                 # Log missing user for feedback
                 logger.warning(f"[Feedback ID: {feedback.id}] User ID {feedback.user_id} not found. Using default avatar.")
                 avatar_path = 'images/default_avatar.png' # Ensure default if user is missing
        else:
            # Log anonymous feedback
            logger.info(f"[Feedback ID: {feedback.id}] Feedback is anonymous. Using default avatar.")
            avatar_path = 'images/default_avatar.png' # Ensure default for anonymous

        testimonials_data.append({
            'user_name': user_name,
            'category': feedback.category,
            'message': feedback.message,
            'rating': feedback.rating,
            'avatar': avatar_path
            # Removed 'source' field
        })

    # Removed shuffling as there's only one data source now

    # Get current date/time for the template
    current_time = datetime.utcnow()

    return render_template(
        'index_refactored.html', 
        cycle_info=cycle_info,
        shared_stories=shared_stories_data, # Pass processed list
        med_reminders=med_reminders,
        testimonials=testimonials_data, # Pass processed list
        username=username,
        now=current_time # Pass current time to the template
    )

@app.route('/assessment')
@login_required
def assessment():
    return render_template('assessment.html')

@app.route('/my_record')
@login_required
def my_record():
    # Get user's assessments
    assessments = Assessment.query.filter_by(
        user_id=session['user_id']
    ).order_by(Assessment.assessment_date.desc()).all()
    
    # Get user's assessments - only those created from the records page
    assessments = Assessment.query.filter_by(
        user_id=session['user_id'],
        assessment_source="records"
    ).order_by(Assessment.assessment_date.desc()).all()
    
    # Get user's medications
    medications = Medication.query.filter_by(
        user_id=session['user_id'], 
        active=True
    ).order_by(Medication.name).all()
    
    # Calendar events for FullCalendar
    calendar_events = []
    
    # Calculate daily averages for pain levels and create calendar events
    daily_pain_data = {}
    hourly_pain_data = {}
    
    for assessment in assessments:
        date_str = assessment.assessment_date.strftime('%Y-%m-%d')
        time_str = f"{assessment.assessment_hour:02d}:00:00"
        datetime_str = f"{date_str}T{time_str}"
        
        # Store hourly assessment data for time-specific display
        hourly_key = f"{date_str}_{assessment.assessment_hour}"
        hourly_pain_data[hourly_key] = {
            'id': assessment.id,
            'pain': assessment.pain_level,
            'hour': assessment.assessment_hour,
            'date': date_str,
            'time': time_str,
            'datetime': datetime_str
        }
        
        # Build or update daily pain data for averages
        if date_str not in daily_pain_data:
            daily_pain_data[date_str] = {
                'total_pain': assessment.pain_level,
                'count': 1,
                'assessments': [assessment.id]
            }
        else:
            daily_pain_data[date_str]['total_pain'] += assessment.pain_level
            daily_pain_data[date_str]['count'] += 1
            daily_pain_data[date_str]['assessments'].append(assessment.id)
    
    # Create calendar events for daily averages
    for date_str, data in daily_pain_data.items():
        avg_pain = data['total_pain'] / data['count']
        
        # Determine color based on average pain level
        if avg_pain <= 3:
            color = '#28a745'  # green for low pain
        elif avg_pain <= 6:
            color = '#ffc107'  # yellow for medium pain
        else:
            color = '#dc3545'  # red for high pain
        
        calendar_events.append({
            'title': f'Avg Pain: {avg_pain:.1f}',
            'start': date_str,
            'color': color,
            'assessment_id': data['assessments'][0],  # Use first assessment for viewing
            'allDay': True,
            'extendedProps': {
                'assessment_id': data['assessments'][0],
                'type': 'average'
            }
        })
    
    # Add hourly events
    for key, data in hourly_pain_data.items():
        # Determine color based on pain level
        pain = data['pain']
        if pain <= 3:
            color = '#28a745'  # green for low pain
        elif pain <= 6:
            color = '#ffc107'  # yellow for medium pain
        else:
            color = '#dc3545'  # red for high pain
        
        # Format time for display in title
        hour = data['hour']
        am_pm = 'AM' if hour < 12 else 'PM'
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        
        calendar_events.append({
            'title': f'{display_hour}{am_pm}: Pain {pain}',
            'start': data['datetime'],
            'color': color,
            'assessment_id': data['id'],
            'allDay': False,
            'extendedProps': {
                'assessment_id': data['id'],
                'type': 'hourly',
                'hour': hour
            }
        })
    
    # Prepare data for pain trends chart
    trends_data = {
        'labels': [],
        'values': []
    }
    
    # Sort dates for the chart
    sorted_dates = sorted(daily_pain_data.keys())
    for date_str in sorted_dates:
        data = daily_pain_data[date_str]
        avg_pain = data['total_pain'] / data['count']
        trends_data['labels'].append(date_str)
        trends_data['values'].append(avg_pain)
    
    return render_template(
        'my_record.html',
        assessments=assessments,
        medications=medications,
        calendar_events=json.dumps(calendar_events),
        trends_data=json.dumps(trends_data)
    )

def generate_text_description(age, menstrual_irregularity, pain_level, 
                            hormone_abnormality, infertility, bmi):
    """Generate a text description from input parameters"""
    description = f"Patient aged {age}, "
    description += "has irregular menstrual cycles, " if menstrual_irregularity else "has regular menstrual cycles, "
    if pain_level >= 8:
        description += "experiences severe chronic pain, "
    elif pain_level >= 5:
        description += "experiences moderate chronic pain, "
    else:
        description += "experiences mild chronic pain, "
    description += "shows hormone level abnormalities, " if hormone_abnormality else "has normal hormone levels, "
    description += "has fertility issues, " if infertility else "has no reported fertility issues, "
    description += f"with a BMI of {bmi}"
    return description

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data for LLM enhancement
        patient_data = request.form.to_dict()
        
        # Get numerical inputs with proper error handling
        try:
            age = float(request.form.get('age', 30))
            pain_level = float(request.form.get('pain_level', 5))
            bmi = float(request.form.get('bmi', 25))
        except ValueError:
            return jsonify({'error': 'Invalid numerical input. Please enter valid numbers for age, pain level, and BMI.'}), 400
        
        # Get categorical inputs (checkboxes)
        menstrual_irregularity = 1.0 if request.form.get('menstrual_irregularity') == 'on' else 0.0
        hormone_abnormality = 1.0 if request.form.get('hormone_abnormality') == 'on' else 0.0
        infertility = 1.0 if request.form.get('infertility') == 'on' else 0.0

        # Scale numerical features
        numerical_features = np.array([[age, pain_level, bmi]])
        scaled_numerical = scaler.transform(numerical_features)
        
        # Combine scaled numerical and categorical features
        tabular_features = np.array([[
            scaled_numerical[0][0],  # scaled age
            menstrual_irregularity,
            scaled_numerical[0][1],  # scaled pain_level
            hormone_abnormality,
            infertility,
            scaled_numerical[0][2]   # scaled bmi
        ]], dtype=np.float32)

        # Process image if provided
        image_array = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        if 'image' in request.files and request.files['image'].filename:
            image_file = request.files['image']
            image_path = "temp_image.jpg"
            try:
                image_file.save(image_path)
                img = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)
                )
                image_array = tf.keras.preprocessing.image.img_to_array(img)
                image_array = image_array / 255.0
                image_array = np.expand_dims(image_array, axis=0)
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

        # Generate and process text
        text_description = generate_text_description(
            age, menstrual_irregularity, pain_level,
            hormone_abnormality, infertility, bmi
        )
        text_input = request.form.get('text', '')
        full_text = f"{text_input} {text_description}"
        
        # Process text exactly as in training
        text_tensor = tf.constant([full_text], dtype=tf.string)

        # Make prediction
        try:
            prediction = combined_model([
                tf.constant(image_array),
                tf.constant(tabular_features),
                text_tensor
            ], training=False)
            
            prediction_score = float(prediction[0][0])
            
            if prediction_score > 0.8:
                stage = "Stage III/IV (Severe)"
            elif prediction_score > 0.6:
                stage = "Stage II (Moderate)"
            elif prediction_score > 0.4:
                stage = "Stage I (Minimal)"
            else:
                stage = "No clear indicators of endometriosis"

            # Create initial prediction result
            prediction_result = {
                'prediction': {
                'stage': stage,
                    'confidence': f"{prediction_score*100:.1f}%"
                },
                'image_analysis': {
                    'processed': True if 'image' in request.files else False,
                    'score': prediction_score
                },
                'educational_information': f"""Based on our analysis:
- Stage Assessment: {stage}
- Confidence Level: {prediction_score*100:.1f}%
- Key Findings: {text_description}

Recommendations:
1. Consult with a gynecologist to discuss these findings
2. Continue monitoring your symptoms
3. Consider keeping a detailed symptom diary
4. Follow up with additional imaging tests if recommended by your doctor""",
                'disclaimer': "This analysis is for informational purposes only and should not replace professional medical advice."
            }
            
            # Enhance prediction with LLM if available
            enhanced_prediction = llm_enhancer.enhance_prediction(patient_data, prediction_result)
            
            # Process educational information to remove any asterisks or bullet points if present
            if enhanced_prediction and 'educational_information' in enhanced_prediction:
                enhanced_prediction['educational_information'] = enhanced_prediction['educational_information']\
                    .replace('*', '')\
                    .replace('- ', '')\
                    .replace('• ', '')\
                    .replace('\n- ', '\n')\
                    .replace('\n• ', '\n')
            
            # Save assessment to database if user is logged in
            if 'user_id' in session:
                try:
                    # Generate a comprehensive report
                    report_content = llm_enhancer.generate_personalized_report(
                        patient_data=patient_data,
                        prediction=enhanced_prediction
                    )
                    
                    # Create a new assessment record
                    user_id = session['user_id']
                    pain_level = int(float(request.form.get('pain_level', 0)))
                    pain_location = request.form.get('pain_location', '')
                    pain_description = request.form.get('symptomDescription', '')
                    
                    # Get symptom data
                    menstrual_irregularity_bool = menstrual_irregularity > 0.5
                    hormone_abnormality_bool = hormone_abnormality > 0.5
                    infertility_bool = infertility > 0.5
                    
                    # Process other symptoms
                    other_symptoms = {}
                    for symptom in ['bloating', 'fatigue', 'nausea', 'headache', 'dizziness', 'mood_swings', 'backpain']:
                        if request.form.get(symptom) == 'on':
                            other_symptoms[symptom] = True
                    
                    # Create new assessment with report content
                    new_assessment = Assessment(
                        user_id=user_id,
                        assessment_date=datetime.utcnow(),
                        pain_level=pain_level,
                        pain_location=pain_location,
                        pain_description=pain_description,
                        menstrual_irregularity=menstrual_irregularity_bool,
                        hormone_abnormality=hormone_abnormality_bool,
                        infertility=infertility_bool,
                        symptoms=json.dumps(other_symptoms) if other_symptoms else None,
                        prediction_result=stage,
                        confidence_score=prediction_score,
                        report_content=report_content,
                        assessment_source="assessment"
                    )
                    
                    db.session.add(new_assessment)
                    db.session.commit()
                    
                    # Add assessment ID to the response
                    enhanced_prediction['assessment_id'] = new_assessment.id
                    
                except Exception as save_error:
                    print(f"Error saving assessment: {str(save_error)}")
                    # Continue with analysis even if saving fails
            
            return jsonify(enhanced_prediction)

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(f"Shapes: image={image_array.shape}, "
                  f"tabular={tabular_features.shape}, "
                  f"text={text_tensor.shape}")
            return jsonify({
                'prediction': {
                    'stage': 'Error',
                    'confidence': '0%'
                },
                'image_analysis': {
                    'processed': False,
                    'score': 0
                },
                'educational_information': 'An error occurred during analysis.',
                'disclaimer': 'Please try again or contact support if the problem persists.'
            }), 500

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert-speech', methods=['POST'])
def convert_speech():
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
                
        audio_file = request.files['audio']
        
        # Generate a unique filename to avoid conflicts
        unique_filename = f"audio_{uuid.uuid4().hex}.webm"
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, unique_filename)
        
        try:
            # Save to the unique path
            audio_file.save(audio_path)
            logger.info(f"Audio saved to temporary file: {audio_path}")
            
            # Check if Groq should be used
            use_groq = request.form.get('use_groq', 'false').lower() == 'true'
            
            if use_groq:
                # Process with Groq
                stt_processor = GroqSTTProcessor()
                transcription = stt_processor.transcribe(audio_path)
                result = {'transcription': transcription}
                response_data = {
                    'text': result.get('transcription', ''),
                    'enhanced': True,
                    'processor': 'Groq AI'
                }
            else:
                # Process with basic STT
                stt_processor = STTProcessor()
                transcription = stt_processor.transcribe(audio_path)
                result = {'transcription': transcription}
                response_data = {
                    'text': result.get('transcription', ''),
                    'enhanced': False
                }
                
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error during speech processing: {str(e)}")
            return jsonify({'error': f"Failed to process audio: {str(e)}"}), 500
        finally:
            # Clean up the temporary file with error handling
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info(f"Temporary audio file removed: {audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary file {audio_path}: {str(cleanup_error)}")
                # Continue execution even if cleanup fails
                
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return jsonify({'error': f"Failed to process audio: {str(e)}"}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        # Get text from request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        voice = data.get('voice', 'Rachel')  # Default voice
        
        # Check if text is not too long (avoid timeouts and large responses)
        if len(text) > 5000:
            # Truncate if too long
            logger.warning(f"Text too long ({len(text)} chars), truncating to 5000 chars")
            text = text[:5000] + "... (text truncated for length)"
        
        # Generate audio from text
        logger.info(f"Generating TTS for {len(text)} chars with voice: {voice}")
        audio_data = tts.generate_audio(text=text, voice=voice)
        
        # Return audio file
        response = make_response(audio_data)
        response.headers['Content-Type'] = 'audio/mpeg'
        response.headers['Content-Disposition'] = 'attachment; filename=speech.mp3'
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return jsonify({'error': f"Failed to generate speech: {str(e)}"}), 500

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        patient_data = request.form.to_dict()
        prediction_data = json.loads(request.form.get('prediction_data', '{}'))
        
        # Generate personalized report content using LLM
        report_content = llm_enhancer.generate_personalized_report(
            patient_data=patient_data,
            prediction=prediction_data
        )
        
        # Process the report to remove any asterisks or bullet points
        if report_content:
            report_content = report_content\
                .replace('*', '')\
                .replace('- ', '')\
                .replace('• ', '')\
                .replace('\n- ', '\n')\
                .replace('\n• ', '\n')
        
        # Return the processed report content
        return jsonify({'report_content': report_content})
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_report/<int:assessment_id>')
@login_required
def download_report_by_id(assessment_id):
    # Get the assessment, ensuring it belongs to the logged-in user
    assessment = Assessment.query.filter_by(
        id=assessment_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    # Get user details
    user = User.query.get(session['user_id'])
    
    try:
        # Get symptoms as a dictionary
        symptoms_dict = json.loads(assessment.symptoms) if assessment.symptoms else {}
        
        # Format symptoms for display
        symptoms_formatted = ""
        if symptoms_dict:
            for key, value in symptoms_dict.items():
                if key == 'other':
                    symptoms_formatted += f"Other: {value}\n"
                else:
                    symptoms_formatted += f"{key.replace('_', ' ').title()}\n"
        
        # Generate prediction text
        prediction_text = assessment.prediction_result or "Assessment results not available"
        confidence = ""
        if assessment.confidence_score:
            confidence = f"(Confidence: {assessment.confidence_score * 100:.1f}%)"
        
        # Process text fields to remove asterisks and bullet points
        pain_description = ""
        if assessment.pain_description:
            pain_description = assessment.pain_description\
                .replace('*', '')\
                .replace('- ', '')\
                .replace('• ', '')\
                .replace('\n- ', '\n')\
                .replace('\n• ', '\n')
                
        notes = ""
        if assessment.notes:
            notes = assessment.notes\
                .replace('*', '')\
                .replace('- ', '')\
                .replace('• ', '')\
                .replace('\n- ', '\n')\
                .replace('\n• ', '\n')
        
        # Generate a new hospital-style report
        report_content = f"""
=======================================================================
                       ENDOMETRICS HEALTH ASSESSMENT
=======================================================================

PATIENT INFORMATION:
-------------------
Name: {user.first_name} {user.last_name}
Patient ID: EM-{user.id:04d}
Date of Birth: {user.date_of_birth.strftime('%B %d, %Y') if user.date_of_birth else 'Not provided'}

ASSESSMENT DETAILS:
------------------
Assessment ID: {assessment.id}
Assessment Date: {assessment.assessment_date.strftime('%B %d, %Y')}
Assessment Time: {assessment.assessment_hour:02d}:00
Report Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}

CLINICAL FINDINGS:
-----------------
PAIN ASSESSMENT:
Pain Level: {assessment.pain_level}/10 {' (Severe)' if assessment.pain_level > 7 else ' (Moderate)' if assessment.pain_level > 3 else ' (Mild)'}
Pain Location: {assessment.pain_location or 'Not specified'}

SYMPTOMS ANALYSIS:
Menstrual Irregularity: {'Present' if assessment.menstrual_irregularity else 'Not reported'}
Hormone Abnormality: {'Present' if assessment.hormone_abnormality else 'Not reported'}
Infertility: {'Present' if assessment.infertility else 'Not reported'}

ADDITIONAL SYMPTOMS:
{symptoms_formatted or 'No additional symptoms reported.'}

ASSESSMENT NOTES:
{pain_description or 'No detailed description provided.'}

DIAGNOSTIC IMPRESSION:
---------------------
{prediction_text} {confidence}

RECOMMENDATIONS:
---------------
1. Follow up with gynecologist to discuss these findings
2. Continue symptom tracking through the Endometrics application
3. Consider keeping a detailed symptom diary between appointments
4. Discuss potential treatment options with your healthcare provider

ADDITIONAL NOTES:
----------------
{notes or 'No additional clinical notes.'}

DISCLAIMER:
----------
This report is generated based on self-reported symptoms and algorithmic analysis.
It should not be considered a medical diagnosis or replace professional medical advice.
Please consult with a qualified healthcare professional for proper diagnosis and treatment.

=======================================================================
                           END OF REPORT
=======================================================================
"""
        
        # Update stored report
        assessment.report_content = report_content
        db.session.commit()
        
        # Create response with the report
        response = make_response(report_content)
        response.headers['Content-Disposition'] = f'attachment; filename=Endometrics_Assessment_Report_{assessment_id}.txt'
        response.headers['Content-Type'] = 'text/plain'
        
        return response
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        flash('Error generating report. Please try again.', 'danger')
        return redirect(url_for('assessment_detail', assessment_id=assessment_id))

def create_tables():
    """Create database tables if they don't exist"""
    try:
        # Check if tables exist by querying a table
        db.session.execute(text("SELECT 1 FROM user LIMIT 1"))
        print("\nDatabase tables already exist, preserving user data...")
        
        # Verify database integrity
        check_database_integrity()
        
        # Check if testimonial table exists, if not create it
        try:
            db.session.execute(text("SELECT 1 FROM testimonial LIMIT 1"))
            print("Testimonial table exists")
        except Exception as e:
            print("Creating testimonial table...")
            Testimonial.__table__.create(db.engine)
            print("Testimonial table created successfully")
        
    except Exception as e:
        # If tables don't exist, create them
        print("\nCreating database tables...")
        db.create_all()
        print("Database tables created successfully!")
    
    # Add sample data
    try:
        add_sample_testimonials()
    except Exception as e:
        print(f"Error adding sample testimonials: {str(e)}")
        
# Add a migration function to handle adding the report_content column
def migrate_add_report_content():
    """Add report_content column to Assessment table if it doesn't exist"""
    with app.app_context():
        try:
            # Check if column exists by querying it
            db.session.execute(text("SELECT report_content FROM assessment LIMIT 1"))
            print("Column report_content already exists")
        except Exception as e:
            # If error occurs, add the column
            db.session.execute(text("ALTER TABLE assessment ADD COLUMN report_content TEXT"))
            db.session.commit()
            print("Added report_content column to assessment table")

# Add a migration function to handle adding the assessment_source column
def migrate_add_assessment_source():
    """Add assessment_source column to Assessment table if it doesn't exist"""
    with app.app_context():
        try:
            # Check if column exists by querying it
            db.session.execute(text("SELECT assessment_source FROM assessment LIMIT 1"))
            print("Column assessment_source already exists")
        except Exception as e:
            # If error occurs, add the column
            db.session.execute(text("ALTER TABLE assessment ADD COLUMN assessment_source VARCHAR(50) DEFAULT 'records'"))
            db.session.commit()
            print("Added assessment_source column to assessment table")
            
            # Set existing assessments to appropriate sources based on where they were likely created
            # Assessments with prediction_result are likely from the assessment page
            db.session.execute(text("UPDATE assessment SET assessment_source = 'assessment' WHERE prediction_result IS NOT NULL AND prediction_result != ''"))
            db.session.commit()
            print("Updated existing assessments with source information")

@app.route('/add_assessment', methods=['GET', 'POST'])
@login_required
def add_assessment():
    if request.method == 'POST':
        user_id = session['user_id']
        
        try:
            # Get form data
            pain_level = int(request.form.get('pain_level', 0))
            pain_location = request.form.get('pain_location', '')
            pain_description = request.form.get('pain_description', '')
            
            # Get symptom checkboxes
            menstrual_irregularity = 'menstrual_irregularity' in request.form
            hormone_abnormality = 'hormone_abnormality' in request.form
            infertility = 'infertility' in request.form
            
            # Get other symptoms as JSON
            other_symptoms = {}
            if request.form.get('has_bloating') == 'on':
                other_symptoms['bloating'] = True
            if request.form.get('has_fatigue') == 'on':
                other_symptoms['fatigue'] = True
            if request.form.get('has_nausea') == 'on':
                other_symptoms['nausea'] = True
            if request.form.get('has_headache') == 'on':
                other_symptoms['headache'] = True
            if request.form.get('other_symptoms'):
                other_symptoms['other'] = request.form.get('other_symptoms')
            
            # Get medications taken
            medications_taken = []
            if request.form.getlist('medications'):
                medications_taken = request.form.getlist('medications')
            
            # Notes
            notes = request.form.get('notes', '')
            
            # Get custom assessment date if provided
            assessment_date = datetime.utcnow()
            if request.form.get('assessment_date'):
                try:
                    assessment_date = datetime.strptime(request.form.get('assessment_date'), '%Y-%m-%d')
                except ValueError:
                    # If date parsing fails, use current date
                    pass
            
            # Get hour if provided
            assessment_hour = 0
            if request.form.get('assessment_hour'):
                try:
                    assessment_hour = int(request.form.get('assessment_hour'))
                    # Ensure hour is between 0-23
                    assessment_hour = max(0, min(23, assessment_hour))
                except ValueError:
                    # If parsing fails, use 0
                    assessment_hour = 0
            
            # Create new assessment
            new_assessment = Assessment(
                user_id=user_id,
                assessment_date=assessment_date,
                assessment_hour=assessment_hour,
                pain_level=pain_level,
                pain_location=pain_location,
                pain_description=pain_description,
                menstrual_irregularity=menstrual_irregularity,
                hormone_abnormality=hormone_abnormality,
                infertility=infertility,
                symptoms=json.dumps(other_symptoms) if other_symptoms else None,
                medications_taken=json.dumps(medications_taken) if medications_taken else None,
                notes=notes,
                assessment_source="records"
            )
            
            db.session.add(new_assessment)
            db.session.commit()
            
            flash('Assessment added successfully!', 'success')
            return redirect(url_for('my_record'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Error adding assessment: {str(e)}")
            flash('Error adding assessment. Please try again.', 'danger')
            return redirect(url_for('add_assessment'))
    
    # For GET requests, fetch user's medications for the form
    medications = Medication.query.filter_by(
        user_id=session['user_id'], 
        active=True
    ).order_by(Medication.name).all()
    
    # Check if a specific date was provided in the request
    selected_date = request.args.get('date')
    if selected_date:
        try:
            # Validate the date format
            selected_date = datetime.strptime(selected_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            # If invalid date, don't use it
            selected_date = None
    
    # Check if a specific hour was provided
    selected_hour = None
    if request.args.get('hour'):
        try:
            selected_hour = int(request.args.get('hour'))
            # Ensure hour is between 0-23
            selected_hour = max(0, min(23, selected_hour))
        except ValueError:
            selected_hour = None
    
    # Get today's date for the date input max attribute
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('add_assessment.html', 
                          medications=medications, 
                          selected_date=selected_date,
                          selected_hour=selected_hour,
                          today_date=today_date)

@app.route('/assessment_detail/<int:assessment_id>')
@login_required
def assessment_detail(assessment_id):
    # Get the assessment, ensuring it belongs to the logged-in user
    assessment = Assessment.query.filter_by(
        id=assessment_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    # Parse JSON fields
    symptoms = json.loads(assessment.symptoms) if assessment.symptoms else {}
    medications_taken = json.loads(assessment.medications_taken) if assessment.medications_taken else []
    
    # Get medication details if medications were taken
    medications = []
    if medications_taken:
        medications = Medication.query.filter(
            Medication.id.in_(medications_taken),
            Medication.user_id == session['user_id']
        ).all()
    
    return render_template(
        'assessment_detail.html',
        assessment=assessment,
        symptoms=symptoms,
        medications=medications
    )

@app.route('/edit_assessment/<int:assessment_id>', methods=['GET', 'POST'])
@login_required
def edit_assessment(assessment_id):
    # Get the assessment, ensuring it belongs to the logged-in user
    assessment = Assessment.query.filter_by(
        id=assessment_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    if request.method == 'POST':
        try:
            # Get form data
            pain_level = int(request.form.get('pain_level', 0))
            pain_location = request.form.get('pain_location', '')
            pain_description = request.form.get('pain_description', '')
            
            # Get symptom checkboxes
            assessment.menstrual_irregularity = 'menstrual_irregularity' in request.form
            assessment.hormone_abnormality = 'hormone_abnormality' in request.form
            assessment.infertility = 'infertility' in request.form
            
            # Get other symptoms as JSON
            other_symptoms = {}
            symptom_values = request.form.getlist('symptoms')
            
            if 'bloating' in symptom_values:
                other_symptoms['bloating'] = True
            if 'fatigue' in symptom_values:
                other_symptoms['fatigue'] = True
            if 'nausea' in symptom_values:
                other_symptoms['nausea'] = True
            if 'headache' in symptom_values:
                other_symptoms['headache'] = True
            if 'dizziness' in symptom_values:
                other_symptoms['dizziness'] = True
            if 'backpain' in symptom_values:
                other_symptoms['backpain'] = True
            if 'mood_swings' in symptom_values:
                other_symptoms['mood_swings'] = True
            if 'constipation' in symptom_values:
                other_symptoms['constipation'] = True
                
            if request.form.get('other_symptoms'):
                other_symptoms['other'] = request.form.get('other_symptoms')
            
            # Get medications taken
            medications_taken = request.form.getlist('medications_taken')
            
            # Notes
            notes = request.form.get('notes', '')
            
            # Update assessment
            assessment.pain_level = pain_level
            assessment.pain_location = pain_location
            assessment.pain_description = pain_description
            assessment.symptoms = json.dumps(other_symptoms) if other_symptoms else None
            assessment.medications_taken = json.dumps(medications_taken) if medications_taken else None
            assessment.notes = notes
            
            db.session.commit()
            
            flash('Assessment updated successfully!', 'success')
            return redirect(url_for('assessment_detail', assessment_id=assessment.id))
            
        except Exception as e:
            db.session.rollback()
            print(f"Error updating assessment: {str(e)}")
            flash('Error updating assessment. Please try again.', 'danger')
            return redirect(url_for('edit_assessment', assessment_id=assessment.id))
    
    # For GET requests
    # Parse JSON fields
    symptoms = json.loads(assessment.symptoms) if assessment.symptoms else {}
    med_ids = json.loads(assessment.medications_taken) if assessment.medications_taken else []
    
    # Get all active medications for the user
    active_medications = Medication.query.filter_by(
        user_id=session['user_id'], 
        active=True
    ).order_by(Medication.name).all()
    
    return render_template(
        'edit_assessment.html',
        assessment=assessment,
        symptoms=symptoms,
        med_ids=med_ids,
        active_medications=active_medications
    )

@app.route('/delete_assessment/<int:assessment_id>', methods=['GET'])
@login_required
def delete_assessment(assessment_id):
    # Get the assessment, ensuring it belongs to the logged-in user
    assessment = Assessment.query.filter_by(
        id=assessment_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    try:
        # Delete the assessment
        db.session.delete(assessment)
        db.session.commit()
        flash('Assessment has been deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting assessment: {str(e)}")
        flash('Error deleting assessment. Please try again.', 'danger')
    
    # Get the referer to determine where to redirect back to
    referer = request.headers.get('Referer', '')
    
    # Check if the request came from the profile page
    if '/profile' in referer:
        return redirect(url_for('profile'))
    else:
        return redirect(url_for('my_record'))

@app.route('/add_medication', methods=['GET', 'POST'])
@login_required
def add_medication():
    if request.method == 'POST':
        user_id = session['user_id']
        
        try:
            # Get form data
            name = request.form.get('name')
            dosage = request.form.get('dosage')
            frequency = request.form.get('frequency')
            
            # Parse dates
            start_date = None
            if request.form.get('start_date'):
                start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
                
            end_date = None
            if request.form.get('end_date'):
                end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d').date()
            
            notes = request.form.get('notes', '')
            
            # Validate required fields
            if not name:
                flash('Medication name is required.', 'danger')
                return redirect(url_for('add_medication'))
            
            # Create new medication
            new_medication = Medication(
                user_id=user_id,
                name=name,
                dosage=dosage,
                frequency=frequency,
                start_date=start_date,
                end_date=end_date,
                notes=notes
            )
            
            db.session.add(new_medication)
            db.session.commit()
            
            flash('Medication added successfully!', 'success')
            return redirect(url_for('my_record'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Error adding medication: {str(e)}")
            flash('Error adding medication. Please try again.', 'danger')
            return redirect(url_for('add_medication'))
    
    return render_template('add_medication.html')

@app.route('/update_medication/<int:medication_id>', methods=['POST'])
@login_required
def update_medication(medication_id):
    medication = Medication.query.filter_by(
        id=medication_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    action = request.form.get('action')
    
    if action == 'deactivate':
        medication.active = False
        db.session.commit()
        flash('Medication marked as inactive.', 'success')
    elif action == 'activate':
        medication.active = True
        db.session.commit()
        flash('Medication marked as active.', 'success')
    
    return redirect(url_for('my_record'))

@app.route('/edit_medication/<int:medication_id>', methods=['POST'])
@login_required
def edit_medication(medication_id):
    # Get the medication, ensuring it belongs to the logged-in user
    medication = Medication.query.filter_by(
        id=medication_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    try:
        # Update medication
        medication.name = request.form.get('name')
        medication.dosage = request.form.get('dosage')
        medication.frequency = request.form.get('frequency')
        medication.notes = request.form.get('notes')
        
        db.session.commit()
        flash('Medication updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"Error updating medication: {str(e)}")
        flash('Error updating medication. Please try again.', 'danger')
    
    return redirect(url_for('my_record'))

@app.route('/delete_medication/<int:medication_id>', methods=['POST'])
@login_required
def delete_medication(medication_id):
    # Get the medication, ensuring it belongs to the logged-in user
    medication = Medication.query.filter_by(
        id=medication_id, 
        user_id=session['user_id']
    ).first_or_404()
    
    try:
        # First delete any MedicationTaken records that reference this medication
        MedicationTaken.query.filter_by(medication_id=medication_id).delete()
        
        # Then delete the medication
        db.session.delete(medication)
        db.session.commit()
        flash('Medication has been deleted.', 'medication-success')
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting medication: {str(e)}")
        flash('Error deleting medication. Please try again.', 'medication-error')
    
    return redirect(url_for('my_record'))

@app.route('/get_medication_reminders')
@login_required
def get_medication_reminders():
    today = datetime.utcnow().date()
    
    # Get user's active medications
    medications = Medication.query.filter_by(
        user_id=session['user_id'],
        active=True
    ).all()
    
    # Get medications already taken today
    taken_today = MedicationTaken.query.filter_by(
        user_id=session['user_id'],
        taken_date=today
    ).all()
    
    # Create a dictionary of medication_id -> times_taken
    taken_med_dict = {record.medication_id: record.times_taken for record in taken_today}
    
    # Filter out medications that have already been taken enough times based on frequency
    reminders = []
    for med in medications:
        times_taken = taken_med_dict.get(med.id, 0)
        required_times = get_required_times(med.frequency)
        
        if times_taken < required_times:
            remaining = required_times - times_taken
            reminders.append({
                'id': med.id,
                'name': med.name,
                'dosage': med.dosage,
                'frequency': med.frequency,
                'times_taken': times_taken,
                'required_times': required_times,
                'remaining': remaining
            })
    
    return jsonify(reminders)

@app.route('/mark_medication_taken', methods=['POST'])
@login_required
def mark_medication_taken():
    medication_id = request.form.get('medication_id')
    
    if not medication_id:
        return jsonify({'success': False, 'message': 'Medication ID is required'}), 400
    
    # Verify the medication belongs to the user
    medication = Medication.query.filter_by(
        id=medication_id,
        user_id=session['user_id']
    ).first()
    
    if not medication:
        return jsonify({'success': False, 'message': 'Medication not found'}), 404
    
    today = datetime.utcnow().date()
    
    # Check if medication was already taken today
    existing_record = MedicationTaken.query.filter_by(
        user_id=session['user_id'],
        medication_id=medication_id,
        taken_date=today
    ).first()
    
    required_times = get_required_times(medication.frequency)
    
    if existing_record:
        # If already recorded today, check if we need to increment the count
        if existing_record.times_taken < required_times:
            existing_record.times_taken += 1
            db.session.commit()
            return jsonify({
                'success': True, 
                'message': f'Medication marked as taken ({existing_record.times_taken} of {required_times})',
                'times_taken': existing_record.times_taken,
                'required_times': required_times
            })
        else:
            return jsonify({
                'success': True, 
                'message': f'Medication already taken {required_times} times today',
                'times_taken': existing_record.times_taken,
                'required_times': required_times
            })
    
    # Create new record
    new_record = MedicationTaken(
        user_id=session['user_id'],
        medication_id=medication_id,
        taken_date=today,
        times_taken=1
    )
    
    try:
        db.session.add(new_record)
        db.session.commit()
        return jsonify({
            'success': True, 
            'message': f'Medication marked as taken (1 of {required_times})',
            'times_taken': 1,
            'required_times': required_times
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error marking medication as taken: {str(e)}")
        return jsonify({'success': False, 'message': 'Error recording medication intake'}), 500

# Helper function to determine required times based on frequency
def get_required_times(frequency):
    """Return the number of times a medication should be taken per day based on its frequency"""
    frequency = frequency.lower() if frequency else ""
    
    if "once" in frequency:
        return 1
    elif "twice" in frequency:
        return 2
    elif "three times" in frequency:
        return 3
    elif "four times" in frequency:
        return 4
    elif "five times" in frequency:
        return 5
    elif "six times" in frequency:
        return 6
    elif "every" in frequency and "hour" in frequency:
        # Handle "every X hours" format
        try:
            hours = int(re.search(r'every (\d+) hour', frequency).group(1))
            return max(1, min(24 // hours, 6))  # Limit to sensible range
        except:
            return 1
    else:
        # Default for frequencies we don't recognize
        return 1

@app.route('/get_user_stories')
@login_required
def get_user_stories():
    # Get all stories for the current user
    user_stories = Story.query.filter_by(user_id=session['user_id']).order_by(Story.created_at.desc()).all()
    
    stories_list = []
    for story in user_stories:
        story_data = story.to_dict()
        # Add user avatar info
        user = User.query.get(story.user_id)
        if user.profile_picture:
            story_data['avatar'] = user.profile_picture
        elif user.avatar_choice > 0:
            story_data['avatar'] = f'avatars/avatar_{user.avatar_choice}.jpg'
        else:
            story_data['avatar'] = 'avatars/avatar_1.jpg'
        
        stories_list.append(story_data)
    
    return jsonify(stories_list)

# Story-related routes
@app.route('/submit_story', methods=['POST'])
@login_required
def submit_story():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    
    user_id = session['user_id']
    content = request.form.get('story_content')
    
    # Basic validation
    if not content:
        return jsonify({'success': False, 'message': 'Story content cannot be empty'}), 400

    allow_sharing = 'share_story' in request.form
    
    # Create a new story
    new_story = Story(
        user_id=user_id,
        content=content,
        allow_sharing=allow_sharing,
        created_at=datetime.utcnow()
    )
    
    try:
        db.session.add(new_story)
        db.session.commit()
        
        # Get user details for the response
        user = User.query.get(user_id)
        story_data = new_story.to_dict() # Use the existing to_dict method
        
        # Manually add avatar info to the dict, as to_dict might not include it
        if user:
            story_data['user_first_name'] = user.first_name if user.first_name else 'Anonymous'
            if user.profile_picture:
                story_data['avatar'] = user.profile_picture
            elif user.avatar_choice > 0:
                story_data['avatar'] = f'avatars/avatar_{user.avatar_choice}.jpg'
            else:
                story_data['avatar'] = 'avatars/avatar_1.jpg'
        else:
            story_data['user_first_name'] = 'Anonymous'
            story_data['avatar'] = 'avatars/avatar_1.jpg'
            
        # Return success JSON with story details
        return jsonify({'success': True, 'story': story_data})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error saving story: {e}")
        # Return error JSON
        return jsonify({'success': False, 'message': 'Database error saving story'}), 500

@app.route('/get_shared_stories')
def get_shared_stories():
    # Get all stories that are allowed to be shared
    shared_stories = Story.query.filter_by(allow_sharing=True).order_by(Story.created_at.desc()).limit(10).all()
    
    stories_list = []
    for story in shared_stories:
        story_data = story.to_dict()
        # Add user avatar info if available
        user = User.query.get(story.user_id)
        if user.profile_picture:
            story_data['avatar'] = user.profile_picture
        elif user.avatar_choice > 0:
            story_data['avatar'] = f'avatars/avatar_{user.avatar_choice}.jpg'
        else:
            story_data['avatar'] = 'avatars/avatar_1.jpg'
        
        stories_list.append(story_data)
    
    return jsonify(stories_list)

@app.route('/edit_story/<int:story_id>', methods=['POST'])
@login_required
def edit_story(story_id):
    try:
        # Get the story, ensuring it belongs to the logged-in user
        story = Story.query.filter_by(
            id=story_id, 
            user_id=session['user_id']
        ).first_or_404()
        
        # Update story content
        story.content = request.form.get('story_content')
        story.allow_sharing = 'allow_sharing' in request.form
        
        db.session.commit()
        flash('Your story has been updated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating story: {str(e)}")
        flash('An error occurred while updating your story. Please try again.', 'danger')
    
    return redirect(url_for('profile'))

@app.route('/delete_story/<int:story_id>', methods=['POST'])
@login_required
def delete_story(story_id):
    try:
        # Get the story, ensuring it belongs to the logged-in user
        story = Story.query.filter_by(
            id=story_id, 
            user_id=session['user_id']
        ).first_or_404()
        
        db.session.delete(story)
        db.session.commit()
        flash('Your story has been deleted successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting story: {str(e)}")
        flash('An error occurred while deleting your story. Please try again.', 'danger')
    
    return redirect(url_for('profile'))

# Feedback routes
@app.route('/feedback')
def feedback():
    """Render the feedback form page"""
    # Redirect to login if not authenticated
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    # Get categories for the dropdown
    categories = [
        'General Feedback', 
        'User Interface', 
        'Feature Request', 
        'Bug Report', 
        'Accessibility Issue',
        'Content & Resources',
        'Medical Information',
        'Community & Support'
    ]
    
    # Check if user is logged in
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    
    return render_template('feedback.html', categories=categories, user=user)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback form submission"""
    try:
        # Get form data
        category = request.form.get('category')
        subject = request.form.get('subject')
        message = request.form.get('message')
        rating = request.form.get('rating')
        
        # Convert rating to integer if provided
        rating_value = int(rating) if rating and rating.isdigit() else None
        
        # Check if user is logged in
        user_id = session.get('user_id')
        
        # Basic validation
        if not category or not subject or not message:
            flash('Please fill out all required fields.', 'warning')
            return redirect(url_for('feedback'))
        
        # Create new feedback entry
        new_feedback = Feedback(
            user_id=user_id,
            category=category,
            subject=subject,
            message=message,
            rating=rating_value
        )
        
        db.session.add(new_feedback)
        db.session.commit()
        
        flash('Thank you for your feedback! We appreciate your input.', 'success')
        return redirect(url_for('feedback'))
        
    except Exception as e:
        db.session.rollback()
        print(f"Error submitting feedback: {str(e)}")
        flash('An error occurred while submitting your feedback. Please try again.', 'danger')
        return redirect(url_for('feedback'))

@app.route('/my_feedback')
@login_required
def my_feedback():
    """View user's own feedback submissions"""
    user_feedback = Feedback.query.filter_by(user_id=session['user_id']).order_by(Feedback.created_at.desc()).all()
    return render_template('my_feedback.html', feedback_list=user_feedback)

@app.route('/edit_feedback/<int:feedback_id>', methods=['POST'])
@login_required
def edit_feedback(feedback_id):
    """Edit existing feedback"""
    try:
        # Get the feedback, ensuring it belongs to the logged-in user
        feedback = Feedback.query.filter_by(
            id=feedback_id, 
            user_id=session['user_id']
        ).first_or_404()
        
        # Update feedback
        feedback.category = request.form.get('category')
        feedback.subject = request.form.get('subject')
        feedback.message = request.form.get('message')
        rating = request.form.get('rating')
        feedback.rating = int(rating) if rating and rating.isdigit() else None
        
        db.session.commit()
        flash('Your feedback has been updated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating feedback: {str(e)}")
        flash('An error occurred while updating your feedback. Please try again.', 'danger')
    
    return redirect(url_for('my_feedback'))

@app.route('/delete_feedback/<int:feedback_id>', methods=['POST'])
@login_required
def delete_feedback(feedback_id):
    """Delete feedback"""
    try:
        # Get the feedback, ensuring it belongs to the logged-in user
        feedback = Feedback.query.filter_by(
            id=feedback_id, 
            user_id=session['user_id']
        ).first_or_404()
        
        db.session.delete(feedback)
        db.session.commit()
        flash('Your feedback has been deleted successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting feedback: {str(e)}")
        flash('An error occurred while deleting your feedback. Please try again.', 'danger')
    
    return redirect(url_for('my_feedback'))

@app.route('/prescription', methods=['GET', 'POST'])
@login_required
def prescription():
    if request.method == 'POST':
        if 'prescription' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
            
        file = request.files['prescription']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
            
        # Define allowed extensions for prescription uploads
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if file and allowed_file(file.filename, allowed_extensions):
            try:
                # Save the file temporarily
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{secure_filename(file.filename)}"
                upload_path = os.path.join(app.static_folder, 'uploads', filename)
                
                # Ensure upload directory exists
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                
                # Save the file
                file.save(upload_path)
                
                return jsonify({
                    'success': True,
                    'filename': filename
                })
                
            except Exception as e:
                app.logger.error(f"Error saving prescription: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error saving prescription. Please try again.'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload a JPG or PNG image file.'
            })
    
    # For GET requests, fetch user's prescriptions
    prescriptions = Prescription.query.filter_by(user_id=session['user_id']).order_by(Prescription.upload_date.desc()).all()
    
    # Get the latest prescription analysis if available
    latest_prescription = prescriptions[0] if prescriptions else None
    latest_analysis = json.loads(latest_prescription.analysis) if latest_prescription else None
    
    return render_template('prescription.html', 
                         prescriptions=prescriptions,
                         show_results=bool(latest_analysis),
                         latest_analysis=latest_analysis,
                         results=latest_analysis)

@app.route('/analyze_prescription', methods=['POST'])
@login_required
def analyze_prescription():
    """Analyze a previously uploaded prescription image"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'success': False, 'error': 'No filename provided'})
            
        filename = data['filename']
        file_path = os.path.join(app.static_folder, 'uploads', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Initialize Vision Analyzer
        vision_analyzer = VisionAnalyzer()
        
        # Try to extract text from image
        extracted_text_result = vision_analyzer.extract_text_from_image(file_path)
        
        # Check if we encountered a rate limit or other API error
        if not extracted_text_result["success"]:
            error_msg = extracted_text_result["error"]
            app.logger.error(f"Text extraction failed: {error_msg}")
            
            # Create a prescription record with the error
            error_analysis = {
                "error": f"Failed to extract text: {error_msg}",
                "image_path": f"/static/uploads/{filename}"
            }
            
            new_prescription = Prescription(
                user_id=session['user_id'],
                filename=filename,
                upload_date=datetime.now(),
                analysis=json.dumps(error_analysis)
            )
            db.session.add(new_prescription)
            db.session.commit()
            
            # Return success but indicate there was an API error
            return jsonify({
                'success': True,
                'has_error': True,
                'analysis': error_analysis
            })
        
        # Get Groq API key from environment
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            error_analysis = {
                "error": "Groq API key not configured",
                "image_path": f"/static/uploads/{filename}"
            }
            
            new_prescription = Prescription(
                user_id=session['user_id'],
                filename=filename,
                upload_date=datetime.now(),
                analysis=json.dumps(error_analysis)
            )
            db.session.add(new_prescription)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'has_error': True,
                'analysis': error_analysis
            })
        
        # Try to analyze prescription using extracted text
        try:
            analyzer = GroqPrescriptionAnalyzer(groq_api_key)
            analysis = analyzer.analyze_prescription(
                f"/static/uploads/{filename}",
                extracted_text_result
            )
            
            # Ensure we have a recommendation
            if not analysis.get('recommendations'):
                analysis['recommendations'] = "No specific recommendations available at this time."
            
            # Add image path to analysis data
            analysis['image_path'] = f"/static/uploads/{filename}"
            
            # Log the number of medications found
            medications = analysis.get('medicines', [])
            app.logger.info(f"Found {len(medications)} medications in prescription")
            
        except Exception as api_error:
            app.logger.error(f"Prescription analysis API error: {str(api_error)}")
            
            # Create analysis with error but include extracted text
            error_analysis = {
                "error": f"Error analyzing prescription: {str(api_error)}",
                "image_path": f"/static/uploads/{filename}",
                "text_blocks": [{"text": extracted_text_result.get("full_text", ""), "confidence": 1.0}]
            }
            
            new_prescription = Prescription(
                user_id=session['user_id'],
                filename=filename,
                upload_date=datetime.now(),
                analysis=json.dumps(error_analysis)
            )
            db.session.add(new_prescription)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'has_error': True,
                'analysis': error_analysis
            })
        
        # Create a new prescription record
        new_prescription = Prescription(
            user_id=session['user_id'],
            filename=filename,
            upload_date=datetime.now(),
            analysis=json.dumps(analysis)
        )
        db.session.add(new_prescription)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        app.logger.error(f"Error analyzing prescription: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error analyzing prescription: {str(e)}'
        })

@app.route('/save_prescription_medicine', methods=['POST'])
@login_required
def save_prescription_medicine():
    """Save a medicine from prescription analysis to user's medication list"""
    user_id = session['user_id']
    
    try:
        # Get form data
        name = request.form.get('name')
        dosage = request.form.get('dosage', '')
        notes = request.form.get('notes', '')
        
        # Get frequency from form
        frequency = request.form.get('frequency', 'As prescribed')
        
        # Get today's date for start date
        start_date = datetime.utcnow().date()
        
        # Validate required fields
        if not name:
            flash('Medication name is required.', 'danger')
            return redirect(url_for('prescription'))
        
        # Log the medication being added
        logger.info(f"Adding medication from prescription: {name}, {dosage}, {frequency}")
        
        # Create new medication
        new_medication = Medication(
            user_id=user_id,
            name=name,
            dosage=dosage,
            frequency=frequency,
            start_date=start_date,
            notes=notes
        )
        
        db.session.add(new_medication)
        db.session.commit()
        
        flash(f'Medication "{name}" saved to your records! <a href="{url_for("my_record")}">View in My Records</a>', 'success')
        
        # Redirect with medication name in URL for toast notification
        return redirect(url_for('prescription', saved_med=name))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding medication from prescription: {str(e)}")
        flash('Error saving medication. Please try again.', 'danger')
    
    return redirect(url_for('prescription'))

# Endometriosis myths and facts for the myth buster
ENDOMETRIOSIS_MYTHS = [
    {
        "myth": "Endometriosis pain is just normal period pain.",
        "fact": "Endometriosis pain is often severe and can affect daily life. It's different from typical menstrual cramps and should be taken seriously."
    },
    {
        "myth": "Getting pregnant cures endometriosis.",
        "fact": "Pregnancy may temporarily suppress symptoms, but it is not a cure. Symptoms often return after pregnancy or breastfeeding ends."
    },
    {
        "myth": "Endometriosis only affects older women.",
        "fact": "Endometriosis can affect people of any age who menstruate, including teenagers and young adults."
    },
    {
        "myth": "Hysterectomy is the only effective treatment for endometriosis.",
        "fact": "While hysterectomy may help some patients, it's not a guaranteed cure. There are many other treatment options including medication, hormone therapy, and minimally invasive surgeries."
    },
    {
        "myth": "Endometriosis is rare.",
        "fact": "Endometriosis affects approximately 1 in 10 women of reproductive age worldwide, making it one of the most common gynecological conditions."
    },
    {
        "myth": "If you have painful periods, you definitely have endometriosis.",
        "fact": "While painful periods can be a symptom of endometriosis, they can also be caused by other conditions. Proper diagnosis requires medical evaluation and often laparoscopic surgery."
    },
    {
        "myth": "You can't get pregnant if you have endometriosis.",
        "fact": "While endometriosis can cause fertility issues in some patients, many people with endometriosis can and do conceive naturally."
    },
    {
        "myth": "Endometriosis always causes obvious symptoms.",
        "fact": "Some people with endometriosis have few or no symptoms, while others experience severe pain. The severity of symptoms doesn't always correlate with the extent of the disease."
    }
]

@app.route('/api/myth', methods=['GET'])
def get_random_myth():
    """Return a random endometriosis myth and fact pair"""
    try:
        # Return a random myth from the predefined list
        random_myth = random.choice(ENDOMETRIOSIS_MYTHS)
        return jsonify(random_myth)
    except Exception as e:
        # Return the first myth if anything goes wrong
        return jsonify(ENDOMETRIOSIS_MYTHS[0])

# Catch-all route to redirect undefined routes to login
@app.route('/<path:undefined_route>')
def catch_all(undefined_route):
    """Catch any undefined routes and redirect to login"""
    return redirect(url_for('login'))

@app.route('/test_groq')
@login_required
def test_groq():
    """Simple test route for Groq API"""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        return jsonify({"error": "API key not available"})
    
    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{
                "role": "user", 
                "content": "Test"
            }],
            "model": "gemma-7b-it",
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify({"success": True, "response": response.json()})
        else:
            return jsonify({"error": f"API Error: {response.status_code}", "details": response.json() if response.text else None})
    
    except Exception as e:
        return jsonify({"error": str(e)})

def clear_user_data():
    """Safely clear all user-related data from the database"""
    try:
        print("\nClearing user data...")
        
        # First, drop all tables to ensure a clean slate
        db.session.execute(text("DROP TABLE IF EXISTS medication_taken"))
        db.session.execute(text("DROP TABLE IF EXISTS medication"))
        db.session.execute(text("DROP TABLE IF EXISTS assessment"))
        db.session.execute(text("DROP TABLE IF EXISTS story"))
        db.session.execute(text("DROP TABLE IF EXISTS feedback"))
        db.session.execute(text("DROP TABLE IF EXISTS user"))
        
        # Commit the drops
        db.session.commit()
        print("Old tables dropped successfully!")
        
    except Exception as e:
        db.session.rollback()
        print(f"Error clearing user data: {str(e)}")
        # Don't raise the error, just continue with table creation
        pass

@app.route('/update_menstrual_cycle', methods=['POST'])
@login_required
def update_menstrual_cycle():
    try:
        # Get form data
        last_period_start = datetime.strptime(request.form.get('last_period_start'), '%Y-%m-%d').date()
        cycle_length = int(request.form.get('cycle_length', 28))
        period_duration = int(request.form.get('period_duration', 5))
        
        # Validate inputs
        if cycle_length < 21 or cycle_length > 35:
            flash('Cycle length should be between 21 and 35 days.', 'danger')
            return redirect(url_for('index'))
            
        if period_duration < 2 or period_duration > 8:
            flash('Period duration should be between 2 and 8 days.', 'danger')
            return redirect(url_for('index'))
        
        # Get or create menstrual cycle record
        menstrual_cycle = MenstrualCycle.query.filter_by(user_id=session['user_id']).first()
        if not menstrual_cycle:
            menstrual_cycle = MenstrualCycle(user_id=session['user_id'])
        
        # Update cycle information
        menstrual_cycle.last_period_start = last_period_start
        menstrual_cycle.cycle_length = cycle_length
        menstrual_cycle.period_duration = period_duration
        
        # Save changes
        db.session.add(menstrual_cycle)
        db.session.commit()
        
        flash('Menstrual cycle information updated successfully!', 'success')
        return redirect(url_for('index'))
        
    except ValueError as e:
        flash('Invalid date format. Please use YYYY-MM-DD format.', 'danger')
        return redirect(url_for('index'))
    except Exception as e:
        db.session.rollback()
        print(f"Error updating menstrual cycle: {str(e)}")
        flash('An error occurred while updating your menstrual cycle information.', 'danger')
        return redirect(url_for('index'))

def migrate_add_menstrual_cycle():
    """Create the menstrual_cycle table if it doesn't exist"""
    with app.app_context():
        try:
            # Check if table exists by querying it
            db.session.execute(text("SELECT 1 FROM menstrual_cycle LIMIT 1"))
            print("Table menstrual_cycle already exists")
        except Exception as e:
            # If error occurs, create the table
            print("Creating menstrual_cycle table...")
            db.create_all()
            print("Created menstrual_cycle table successfully")

@app.route('/get_phase_motivation', methods=['POST'])
def get_phase_motivation():
    try:
        # Get current phase from request
        data = request.json
        phase = data.get('phase', 'Unknown')
        
        # Get timestamp if provided (used for cache busting on frontend)
        timestamp = data.get('timestamp')
        
        print(f"Received request for phase: {phase} at timestamp: {timestamp}")
        
        if phase == 'Unknown':
            return jsonify({
                'success': False,
                'message': 'Invalid phase provided'
            })
        
        # Always use Groq to generate phase-specific content
        llm_enhancer = LlamaEnhancer()
        
        # Create prompts based on the specific phase
        if phase == 'Menstrual Phase':
            prompt = """Generate a supportive and empowering quote (2-3 sentences) for someone in their menstrual phase 
            who might be experiencing cramps, fatigue, or discomfort. The tone should be gentle, validating, and encouraging. 
            
            Then provide exactly 4 practical self-care tips specific to the menstrual phase. Format each tip as a complete sentence.
            Format your response with the quote first, followed by a blank line, then the 4 numbered tips (1. First tip, etc.)"""
        elif phase == 'Follicular Phase':
            prompt = """Generate an energizing, forward-looking quote (2-3 sentences) for someone in their follicular phase 
            when energy is rising and they're entering a creative, productive time. The tone should be uplifting and motivating.
            
            Then provide exactly 4 practical tips for maximizing the natural energy of this phase. Format each tip as a complete sentence.
            Format your response with the quote first, followed by a blank line, then the 4 numbered tips (1. First tip, etc.)"""
        elif phase == 'Ovulation Phase':
            prompt = """Generate an empowering and confident quote (2-3 sentences) for someone in their ovulation phase 
            when they're at peak energy, confidence, and social ability. The tone should be bold and affirming.
            
            Then provide exactly 4 practical tips for channeling the high energy of this phase. Format each tip as a complete sentence.
            Format your response with the quote first, followed by a blank line, then the 4 numbered tips (1. First tip, etc.)"""
        elif phase == 'Luteal Phase':
            prompt = """Generate a calming, nurturing quote (2-3 sentences) for someone in their luteal phase 
            who might be experiencing PMS symptoms and decreasing energy. The tone should be soothing and reassuring.
            
            Then provide exactly 4 practical self-care tips specific to managing luteal phase symptoms. Format each tip as a complete sentence.
            Format your response with the quote first, followed by a blank line, then the 4 numbered tips (1. First tip, etc.)"""
        else:
            prompt = """Generate an uplifting wellness quote (2-3 sentences) related to female health and self-care.
            
            Then provide exactly 4 general wellness tips that support women's health. Format each tip as a complete sentence.
            Format your response with the quote first, followed by a blank line, then the 4 numbered tips (1. First tip, etc.)"""
            
        print(f"Sending prompt for phase: {phase}")
        
        # Use a higher temperature for more variety in responses
        response = llm_enhancer.get_response(
            prompt,
            temperature=0.9,  # Increased from 0.7 to get more variety
            max_tokens=350
        )
        
        print(f"Raw response received: {response}")
        
        # Extract quote and tips from the response
        try:
            # Parse the response - we expect the quote first, followed by tips
            lines = response.strip().split('\n')
            
            print(f"Split into {len(lines)} lines: {lines}")
            
            # The first non-empty paragraph should be the quote
            quote = ""
            tips = []
            
            quote_found = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if not quote_found:
                    # This is the quote
                    quote = line
                    quote_found = True
                    print(f"Found quote: {quote}")
                elif line.startswith("1.") or line.startswith("- ") or line.startswith("•"):
                    # This is the start of tips
                    tip_text = line.lstrip("1.- •").strip()
                    tips.append(tip_text)
                    print(f"Found tip 1: {tip_text}")
                elif line.startswith("2.") or line.startswith("3.") or line.startswith("4."):
                    tip_text = line.lstrip("234.- •").strip()
                    tips.append(tip_text)
                    print(f"Found additional tip: {tip_text}")
            
            # If parsing failed, set fallback content
            if not quote or len(tips) < 2:
                print(f"Parsing failed. Quote found: {quote}, Tips count: {len(tips)}")
                raise ValueError("Failed to parse response properly")
                
            # Ensure we have exactly 4 tips
            while len(tips) < 4:
                if phase == 'Menstrual Phase':
                    tips.append("Practice gentle self-care and prioritize rest")
                elif phase == 'Follicular Phase':
                    tips.append("Harness your creative energy for new projects")
                elif phase == 'Ovulation Phase':
                    tips.append("Connect with others while your social energy is high")
                elif phase == 'Luteal Phase':
                    tips.append("Be gentle with yourself as energy decreases")
                else:
                    tips.append("Practice mindful wellness routines daily")
                
                print(f"Added fallback tip, now have {len(tips)} tips")
                    
            # Limit to max 4 tips
            tips = tips[:4]
            
            print(f"Final response - Quote: '{quote}', Tips: {tips}")
            
            return jsonify({
                'success': True,
                'quote': f'"{quote}"',
                'tips': tips,
                'timestamp': timestamp  # Return the timestamp for debugging
            })
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            # Fall back to static content
            fallback_quote = get_phase_quote(phase)
            fallback_tips = get_phase_tips(phase.lower().split()[0])[:4]  # Get first 4 tips
            
            return jsonify({
                'success': True,
                'quote': f'"{fallback_quote}"',
                'tips': fallback_tips,
                'is_fallback': True,
                'timestamp': timestamp  # Return the timestamp for debugging
            })
    
    except Exception as e:
        print(f"Error generating phase motivation: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error: {str(e)}"
        })

def get_cycle_info(user_id):
    cycle = MenstrualCycle.query.filter_by(user_id=user_id).order_by(MenstrualCycle.updated_at.desc()).first()
    
    if not cycle:
        return None
    
    current_date = datetime.utcnow().date()
    current_phase = cycle.get_current_phase()
    next_phase = cycle.get_next_phase()
    
    return {
        'last_period_start': cycle.last_period_start,
        'cycle_length': cycle.cycle_length,
        'period_duration': cycle.period_duration,
        'current_phase': current_phase['phase'],
        'days_in_phase': current_phase['days'],
        'next_phase': next_phase['phase'],
        'next_phase_date': next_phase['date'],
        'today_date': current_date
    }

def get_phase_quote(phase):
    """Get a motivational quote specific to the current cycle phase"""
    quotes = {
        'Menstrual Phase': "Rest and reflection are essential during this phase. Your body is working hard, and it's okay to slow down and honor its needs.",
        'Follicular Phase': "This is a time of renewed energy and growth. Embrace your creativity and set intentions for the cycle ahead.",
        'Ovulation Phase': "You're at your peak energy! This is a great time for challenging activities, socializing, and accomplishing important tasks.",
        'Luteal Phase': "Practice self-compassion as your body prepares for menstruation. Listen to your intuition and embrace the wisdom it offers."
    }
    
    return quotes.get(phase, "Honor your body's natural rhythms and trust in its wisdom, no matter where you are in your cycle.")

def get_phase_tips(phase):
    """Return tips based on the current menstrual cycle phase"""
    tips = {
        'menstrual': [
            "Make sure to stay hydrated by drinking plenty of water.",
            "Gentle exercise like walking or yoga can help ease discomfort.",
            "Apply a heating pad to your abdomen to relieve cramps.",
            "Consider taking anti-inflammatory medications if approved by your doctor.",
            "Practice self-care and allow yourself rest when needed."
        ],
        'follicular': [
            "This is a great time for more intense workouts as your energy may be higher.",
            "Focus on iron-rich foods to replenish what was lost during menstruation.",
            "Schedule important meetings or events during this time for peak mental clarity.",
            "Start new projects now while your creative energy is on the rise.",
            "Your skin may be clearer now, a good time for special events."
        ],
        'ovulatory': [
            "You may notice increased energy and confidence during this phase.",
            "This is a good time for social activities as you may feel more outgoing.",
            "Consider tracking fertility signs if planning or avoiding pregnancy.",
            "Stay hydrated and maintain balanced nutrition.",
            "Some women experience slight pain during ovulation; mild pain relievers can help."
        ],
        'luteal': [
            "You might experience food cravings; keep healthy snacks on hand.",
            "Mood swings may occur; practice mindfulness or meditation.",
            "Reduce caffeine and sugar to minimize PMS symptoms.",
            "Try to get extra rest as your energy levels might decrease.",
            "Incorporate calcium-rich foods to help reduce PMS symptoms."
        ]
    }
    
    return tips.get(phase, ["Focus on regular exercise and balanced nutrition.",
                           "Practice stress-reducing activities like meditation.",
                           "Maintain a consistent sleep schedule.",
                           "Stay hydrated throughout the day.",
                           "Connect with supportive friends and family."])

def add_sample_testimonials():
    """Add sample testimonials to the database if none exist"""
    try:
        # Check if testimonials table exists
        db.session.execute(text("SELECT 1 FROM testimonial LIMIT 1"))
        
        # Only add if no testimonials exist
        if Testimonial.query.count() == 0:
            sample_testimonials = [
                Testimonial(
                    name="Sarah Johnson",
                    content="Endometrics has been a game-changer for my health journey. The personalized insights and tracking features have helped me understand my body better than ever before.",
                    rating=5,
                    avatar="https://randomuser.me/api/portraits/women/1.jpg"
                ),
                Testimonial(
                    name="Michael Chen",
                    content="As someone who struggled with health tracking, Endometrics made it simple and intuitive. The AI-powered analysis is incredibly accurate and helpful.",
                    rating=5,
                    avatar="https://randomuser.me/api/portraits/men/1.jpg"
                ),
                Testimonial(
                    name="Emily Rodriguez",
                    content="The menstrual cycle tracking feature is amazing! It's helped me predict my cycles with incredible accuracy and manage my symptoms better.",
                    rating=4,
                    avatar="https://randomuser.me/api/portraits/women/2.jpg"
                )
            ]
            
            for testimonial in sample_testimonials:
                db.session.add(testimonial)
            
            db.session.commit()
            print("Sample testimonials added successfully!")
        else:
            print("Testimonials already exist in database")
    except Exception as e:
        print(f"Error in add_sample_testimonials: {str(e)}")
        db.session.rollback()

# Testimonial Routes
@app.route('/submit_testimonial', methods=['POST'])
@login_required
def submit_testimonial():
    """Submit a new testimonial"""
    try:
        user = User.query.get(session['user_id'])
        content = request.form.get('content')
        rating = request.form.get('rating')
        display_name = request.form.get('display_name', '')
        
        # Convert rating to integer
        rating_value = int(rating) if rating and rating.isdigit() else 5
        
        # Use the user's avatar or default
        avatar = 'images/default_avatar.png' # Corrected default path
        if user.profile_picture:
            avatar = user.profile_picture
        elif user.avatar_choice > 0:
            avatar = f'avatars/avatar_{user.avatar_choice}.jpg'
        
        # Create new testimonial
        new_testimonial = Testimonial(
            name=display_name or None,  # Use None if empty to display as Anonymous
            content=content,
            rating=rating_value,
            avatar=avatar
        )
        
        db.session.add(new_testimonial)
        db.session.commit()
        
        flash('Thank you for your testimonial! It will be reviewed and published soon.', 'success')
        
    except Exception as e:
        db.session.rollback()
        print(f"Error submitting testimonial: {str(e)}")
        flash('An error occurred while submitting your testimonial. Please try again.', 'danger')
    
    return redirect(url_for('index'))

@app.route('/get_testimonials')
def get_testimonials():
    """Get testimonials for the website"""
    testimonials = Testimonial.query.order_by(func.random()).limit(6).all()
    testimonials_list = [t.to_dict() for t in testimonials]
    return jsonify(testimonials_list)

def migrate_add_prescription():
    """Add prescription table if it doesn't exist"""
    try:
        with app.app_context():
            # Check if table exists
            inspector = inspect(db.engine)
            if 'prescription' not in inspector.get_table_names():
                print("Creating prescription table...")
                # Create only the prescription table
                Prescription.__table__.create(db.engine)
                print("Prescription table created successfully")
            else:
                print("Prescription table already exists")
    except Exception as e:
        print(f"Error creating prescription table: {str(e)}")

# Route to handle bulk feedback deletion
@app.route('/delete_selected_feedback', methods=['POST'])
@login_required
def delete_selected_feedback():
    try:
        data = request.get_json()
        feedback_ids = data.get('feedback_ids', [])
        user_id = session['user_id']
        
        if not feedback_ids:
            return jsonify({'success': False, 'message': 'No feedback IDs provided.'}), 400
        
        # Query feedback items belonging to the user
        feedbacks_to_delete = Feedback.query.filter(
            Feedback.id.in_(feedback_ids),
            Feedback.user_id == user_id
        ).all()
        
        if not feedbacks_to_delete:
            return jsonify({'success': False, 'message': 'No matching feedback found for this user.'}), 404
            
        deleted_count = 0
        for feedback in feedbacks_to_delete:
            db.session.delete(feedback)
            deleted_count += 1
            
        db.session.commit()
        logger.info(f"User {user_id} deleted {deleted_count} feedback items.")
        return jsonify({'success': True, 'message': f'{deleted_count} feedback items deleted.'})    
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting selected feedback for user {session.get('user_id')}: {str(e)}")
        return jsonify({'success': False, 'message': 'An internal error occurred.'}), 500

# Route to handle bulk story deletion
@app.route('/delete_selected_stories', methods=['POST'])
@login_required
def delete_selected_stories():
    try:
        data = request.get_json()
        story_ids = data.get('story_ids', [])
        user_id = session['user_id']
        
        if not story_ids:
            return jsonify({'success': False, 'message': 'No story IDs provided.'}), 400
            
        # Query stories belonging to the user
        stories_to_delete = Story.query.filter(
            Story.id.in_(story_ids),
            Story.user_id == user_id
        ).all()
        
        if not stories_to_delete:
            return jsonify({'success': False, 'message': 'No matching stories found for this user.'}), 404
            
        deleted_count = 0
        for story in stories_to_delete:
            db.session.delete(story)
            deleted_count += 1
            
        db.session.commit()
        logger.info(f"User {user_id} deleted {deleted_count} stories.")
        return jsonify({'success': True, 'message': f'{deleted_count} stories deleted.'})    
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting selected stories for user {session.get('user_id')}: {str(e)}")
        return jsonify({'success': False, 'message': 'An internal error occurred.'}), 500

def migrate_add_medication_taken_counts():
    """Add the times_taken and updated_at columns to the MedicationTaken table"""
    try:
        # Check if the column already exists
        result = db.session.execute(text("PRAGMA table_info(medication_taken)")).fetchall()
        columns = [row[1] for row in result]
        
        changes_made = False
        
        # Add times_taken column if it doesn't exist
        if 'times_taken' not in columns:
            db.session.execute(text("ALTER TABLE medication_taken ADD COLUMN times_taken INTEGER DEFAULT 1"))
            changes_made = True
            logger.info("Added times_taken column to MedicationTaken table")
        
        # Add updated_at column if it doesn't exist
        if 'updated_at' not in columns:
            db.session.execute(text("ALTER TABLE medication_taken ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            changes_made = True
            logger.info("Added updated_at column to MedicationTaken table")
        
        if changes_made:
            db.session.commit()
            logger.info("MedicationTaken table migration completed successfully")
        else:
            logger.info("MedicationTaken table already has the required columns")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in migrate_add_medication_taken_counts: {str(e)}")

# Run all migrations at startup
def run_migrations():
    """Run all database migrations"""
    migrate_add_report_content()
    migrate_add_assessment_source()
    migrate_add_menstrual_cycle()
    migrate_add_prescription()
    migrate_add_medication_taken_counts()
    logger.info("All migrations completed")

class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    analysis = db.Column(db.Text)  # Store the analysis results as JSON string
    
    # Relationship with User
    user = db.relationship('User', backref='prescriptions', lazy=True)
    
    def __repr__(self):
        return f'<Prescription {self.id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': json.loads(self.analysis) if self.analysis else {}
        }

if __name__ == '__main__':
    # Create an application context before performing database operations
    with app.app_context():
        # Create tables
        create_tables()
        
        # Run migrations
        run_migrations()
        
        # Ensure static folders exist
        ensure_static_files()
        
        # Add sample testimonials if needed
        add_sample_testimonials()
    
    # Start server
    app.run(debug=True, host='0.0.0.0')