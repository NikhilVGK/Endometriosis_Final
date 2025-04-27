import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate, TextVectorization, Embedding, LSTM
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf


# Define image size
IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
TABULAR_FEATURES = 6 # Number of features from CSV

# Add these constants after the existing ones
MAX_WORDS = 1000  # Maximum number of words to keep
MAX_LENGTH = 100  # Maximum length of each text input
EMBEDDING_DIM = 64  # Dimension of word embeddings


# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Rescale pixel values
        return img_array
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: Error loading image {image_path}: {e}. Skipping.")
        return None

def generate_text_description(row):
    """Generate a text description from tabular data"""
    description = f"Patient aged {row['Age']}, "
    
    # Add menstrual irregularity info
    description += "has irregular menstrual cycles, " if row['Menstrual_Irregularity'] else "has regular menstrual cycles, "
    
    # Add pain level info
    pain_level = row['Chronic_Pain_Level']
    if pain_level >= 8:
        description += "experiences severe chronic pain, "
    elif pain_level >= 5:
        description += "experiences moderate chronic pain, "
    else:
        description += "experiences mild chronic pain, "
    
    # Add hormone level info
    description += "shows hormone level abnormalities, " if row['Hormone_Level_Abnormality'] else "has normal hormone levels, "
    
    # Add fertility info
    description += "has fertility issues, " if row['Infertility'] else "has no reported fertility issues, "
    
    # Add BMI info
    description += f"with a BMI of {row['BMI']}"
    
    return description

def load_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    images = []
    tabular_data = []
    text_data = []  # Add this list for text data
    labels = []

    # --- Modification Start: Get actual filenames --- 
    neg_dir = os.path.join(img_dir, 'negative')
    pos_dir = os.path.join(img_dir, 'positive')

    try:
        # Get sorted list of filenames for each class
        neg_filenames = sorted([f for f in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, f))])
        pos_filenames = sorted([f for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error accessing image directories: {e}. Ensure '{neg_dir}' and '{pos_dir}' exist.") from e

    print(f"Found {len(neg_filenames)} images in {neg_dir}")
    print(f"Found {len(pos_filenames)} images in {pos_dir}")
    # --- Modification End --- 

    # Keep track of counts for indexing into filename lists
    neg_count = 0
    pos_count = 0

    # Define numerical features for scaling
    numerical_cols = ['Age', 'Chronic_Pain_Level', 'BMI']
    categorical_cols = ['Menstrual_Irregularity', 'Hormone_Level_Abnormality', 'Infertility']

    # Fit scaler on the entire numerical data first
    scaler = StandardScaler()
    # Handle potential non-numeric values before scaling
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numerical_cols, inplace=True)
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Iterate through the dataframe
    for index, row in df.iterrows():
        label = int(row['Diagnosis'])
        
        # Generate text description for this patient
        text_description = generate_text_description(row)
        
        # --- Modification Start: Use actual filenames based on order --- 
        # WARNING: This assumes the order of files in the directory matches the order
        #          of samples for that class in the CSV. This might be unreliable.
        img_filename = None
        img_path = None
        if label == 0:
            if neg_count < len(neg_filenames):
                img_filename = neg_filenames[neg_count]
                img_path = os.path.join(neg_dir, img_filename)
                neg_count += 1
            else:
                print(f"Warning: Ran out of negative images before processing all CSV rows (index {neg_count}). Skipping row {index}.")
                continue
        else:
            if pos_count < len(pos_filenames):
                img_filename = pos_filenames[pos_count]
                img_path = os.path.join(pos_dir, img_filename)
                pos_count += 1
            else:
                print(f"Warning: Ran out of positive images before processing all CSV rows (index {pos_count}). Skipping row {index}.")
                continue
        # --- Modification End --- 

        # Load and preprocess image
        img_array = load_and_preprocess_image(img_path)

        if img_array is not None:
            # Extract, scale numerical, and combine tabular features
            tab_features = row[numerical_cols + categorical_cols].values.astype('float32')

            images.append(img_array)
            tabular_data.append(tab_features)
            text_data.append(text_description)  # Add the generated text description
            labels.append(label)

    if not images: # Handle case where no images were successfully loaded
        # Check if the counts match the number of files found initially
        if neg_count == 0 and pos_count == 0 and (len(neg_filenames) > 0 or len(pos_filenames) > 0):
             raise ValueError("No images were loaded successfully. Check image file integrity, formats, or loading errors.")
        else:
            raise ValueError("No images could be loaded. Check image paths, naming convention, or if directories are empty.")

    print(f"Successfully loaded {len(images)} images and corresponding tabular data.")
    print(f"Negative samples processed: {neg_count}, Positive samples processed: {pos_count}")

    # Convert lists to numpy arrays
    images = np.array(images)
    tabular_data = np.array(tabular_data)
    text_data = np.array(text_data)  # Convert text data to numpy array
    labels = np.array(labels) # Labels are already 0 or 1

    # Split all data including text
    X_train_img, X_val_img, X_train_tab, X_val_tab, X_train_text, X_val_text, y_train, y_val = train_test_split(
        images, tabular_data, text_data, labels, 
        test_size=0.5, random_state=42, stratify=labels
    )

    return X_train_img, X_val_img, X_train_tab, X_val_tab, X_train_text, X_val_text, y_train, y_val


# Build CNN model for images (adjusted input shape)
def build_image_model(input_shape):
    image_input = Input(shape=input_shape, name='image_input')
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    # Output layer before concatenation (can be adjusted)
    image_features = Dense(128, activation='relu', name='image_features')(x) 
    # Return a Model, not Sequential, for easier feature extraction/combination
    model = Model(inputs=image_input, outputs=image_features) 
    return model

# Build model for tabular data (updated input shape and output)
def build_tabular_model(input_shape): # Input shape is now number of features
    tabular_input = Input(shape=(input_shape,), name='tabular_input')
    x = Dense(64, activation='relu')(tabular_input) # Adjusted layer size
    x = Dropout(0.5)(x)
    tabular_features = Dense(32, activation='relu', name='tabular_features')(x) # Adjusted layer size
    # Return a Model
    model = Model(inputs=tabular_input, outputs=tabular_features)
    return model

# Add this new function for text model
def build_text_model(max_words=MAX_WORDS, max_length=MAX_LENGTH, embedding_dim=EMBEDDING_DIM):
    text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
    
    # Text vectorization layer
    vectorize_layer = TextVectorization(
        max_tokens=max_words,
        output_mode='int',
        output_sequence_length=max_length,
        standardize='lower_and_strip_punctuation'
    )
    
    # Create embedding and LSTM layers
    x = vectorize_layer(text_input)
    x = Embedding(max_words + 1, embedding_dim)(x)  # +1 for padding token
    x = LSTM(32)(x)
    text_features = Dense(32, activation='relu', name='text_features')(x)
    
    return Model(inputs=text_input, outputs=text_features), vectorize_layer

# Modify the build_combined_model function
def build_combined_model(image_model, tabular_model, text_model):
    # Get the output tensors from the individual models
    image_features = image_model.output
    tabular_features = tabular_model.output
    text_features = text_model.output

    # Concatenate all features
    combined_features = concatenate([image_features, tabular_features, text_features])

    # Add final classification layers
    x = Dense(64, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='combined_output')(x)

    # Create the combined model with all three inputs
    model = Model(
        inputs=[image_model.input, tabular_model.input, text_model.input],
        outputs=output
    )
    return model

def train_models():
    # Load data using the modified function
    print("Loading and preprocessing data...")
    X_train_img, X_val_img, X_train_tab, X_val_tab, X_train_text, X_val_text, y_train, y_val = load_data(
        'data/symptoms.csv', 'data/images'
    )

    # Build models
    print("Building models...")
    image_model_base = build_image_model(IMG_SHAPE)
    tabular_model_base = build_tabular_model(TABULAR_FEATURES)
    text_model_base, vectorize_layer = build_text_model()
    
    # Convert text data to string tensors and adapt vectorization layer
    X_train_text = tf.convert_to_tensor(X_train_text, dtype=tf.string)
    X_val_text = tf.convert_to_tensor(X_val_text, dtype=tf.string)
    vectorize_layer.adapt(X_train_text)
    
    combined_model = build_combined_model(image_model_base, tabular_model_base, text_model_base)

    # Compile combined model
    print("Compiling model...")
    combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train combined model
    print("Starting training...")
    history = combined_model.fit(
        [X_train_img, X_train_tab, X_train_text],
        y_train,
        validation_data=([X_val_img, X_val_tab, X_val_text], y_val),
        epochs=10,
        batch_size=32
    )

    # Save the combined model and vectorize layer
    print("Saving models...")
    if not os.path.exists('models'):
        os.makedirs('models')
    combined_model.save('models/combined_model.h5')
    
    # Save the vectorize layer configuration and vocabulary
    vectorize_config = {
        'config': vectorize_layer.get_config(),
        'vocabulary': vectorize_layer.get_vocabulary()
    }
    with open('models/vectorize_config.pkl', 'wb') as f:
        pickle.dump(vectorize_config, f)
    
    print("Training finished and models saved.")


if __name__ == '__main__':
    train_models()