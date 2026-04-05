import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import random

# MediaPipe 0.10.31+ dropped the legacy Python "solutions" API (Holistic, etc.).
# This project still uses mp.solutions.holistic — pin an older wheel, e.g.:
#   pip install "mediapipe>=0.10.9,<0.10.31"
if not hasattr(mp, "solutions"):
    raise ImportError(
        "mediapipe has no attribute 'solutions' (too new a mediapipe build for this script). "
        'Install a legacy-compatible version, e.g.: pip install "mediapipe>=0.10.9,<0.10.31"'
    )
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout,
    Concatenate, TimeDistributed, BatchNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D, GaussianNoise, 
    SpatialDropout1D, GlobalAveragePooling2D
)

# ==========================================
# 0. GPU & SYSTEM CONFIGURATION
# ==========================================
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth Enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Changed to the root folder to capture all subcategories (Adjectives, Animals, etc.)
DATASET_DIR = "Include50/" 
FEATURES_DIR = "Extracted_Features_Full/"  
IMG_SIZE = 224
MAX_FRAMES = 25
BATCH_SIZE = 32  # Reduced slightly for higher class counts
EPOCHS = 100
SEED = 42

mp_holistic = mp.solutions.holistic
CNN_DIM = 1280
MP_DIM = 1662 
COMBINED_DIM = CNN_DIM + MP_DIM 

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 2. DATASET CRAWLING & FEATURE EXTRACTION
# ==========================================

def get_all_sign_folders(root_dir):
    """
    Recursively finds all folders that contain video files.
    Returns a list of absolute paths to folders containing signs.
    """
    sign_folders = []
    valid_ext = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    for root, dirs, files in os.walk(root_dir):
        # Check if this specific folder contains any video files
        if any(f.lower().endswith(valid_ext) for f in files):
            sign_folders.append(root)
            
    return sorted(sign_folders)

def extract_mediapipe_landmarks(results):
    """Extracts landmarks and zero-centers relative to the nose for spatial invariance."""
    base_x, base_y, base_z = 0.0, 0.0, 0.0
    if results.pose_landmarks and len(results.pose_landmarks.landmark) > 0:
        nose = results.pose_landmarks.landmark[0]
        base_x, base_y, base_z = nose.x, nose.y, nose.z

    def get_coords(landmarks, num_pts, dim=3):
        if landmarks:
            coords = []
            for res in landmarks.landmark:
                coords.extend([res.x - base_x, res.y - base_y, res.z - base_z])
                if dim == 4: coords.append(getattr(res, 'visibility', 0.0))
            return np.array(coords)
        return np.zeros(num_pts * dim)

    pose = get_coords(results.pose_landmarks, 33, 4) 
    face = get_coords(results.face_landmarks, 468, 3) 
    lh = get_coords(results.left_hand_landmarks, 21, 3) 
    rh = get_coords(results.right_hand_landmarks, 21, 3) 
    return np.concatenate([pose, face, lh, rh])

def pre_extract_full_dataset(root_dir, features_dir):
    """Processes all categories and saves .npy files in a mirrored structure."""
    sign_folders = get_all_sign_folders(root_dir)
    print(f"🔍 Found {len(sign_folders)} distinct sign classes.")
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    cnn_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        for folder_path in sign_folders:
            # Create a relative path for the output folder
            rel_path = os.path.relpath(folder_path, root_dir)
            out_cls_dir = os.path.join(features_dir, rel_path)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            for file in os.listdir(folder_path):
                if not file.lower().endswith(valid_extensions): continue
                
                vid_path = os.path.join(folder_path, file)
                npy_path = os.path.join(out_cls_dir, file + ".npy")
                
                if os.path.exists(npy_path): continue
                
                print(f"Processing: {rel_path} -> {file}")
                cap = cv2.VideoCapture(vid_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                skip_interval = max(int(total_frames / MAX_FRAMES), 1)
                
                features = []
                while cap.isOpened() and len(features) < MAX_FRAMES:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if len(features) * skip_interval <= cap.get(cv2.CAP_PROP_POS_FRAMES):
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cnn_feat = cnn_extractor.predict(np.expand_dims(cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255.0, 0), verbose=0)[0]
                        mp_feat = extract_mediapipe_landmarks(holistic.process(img))
                        features.append(np.concatenate([cnn_feat, mp_feat]))
                cap.release()
                
                while len(features) < MAX_FRAMES: features.append(np.zeros(COMBINED_DIM))
                np.save(npy_path, np.array(features))

# ==========================================
# 3. LOAD DATASET
# ==========================================
def load_full_npy_dataset(features_dir):
    X, y = [], []
    sign_folders = []
    # Find all subdirectories that contain .npy files
    for root, dirs, files in os.walk(features_dir):
        if any(f.endswith('.npy') for f in files):
            sign_folders.append(root)
            
    sign_folders = sorted(sign_folders)
    # Use the leaf folder name as the label
    class_mapping = {os.path.basename(path): i for i, path in enumerate(sign_folders)}
    
    for path in sign_folders:
        cls_name = os.path.basename(path)
        for file in os.listdir(path):
            if file.endswith('.npy'):
                X.append(np.load(os.path.join(path, file)))
                y.append(class_mapping[cls_name])
                
    return np.array(X), np.array(y), class_mapping

# ==========================================
# 4. MODEL & INFERENCE
# ==========================================
def build_robust_isl_model(max_frames, combined_dim, num_classes):
    inputs = Input(shape=(max_frames, combined_dim))
    x = GaussianNoise(0.01)(inputs)
    
    # Projection/Bottleneck
    cnn_p = TimeDistributed(Dense(256, activation='swish'))(x[:, :, :1280])
    mp_p = TimeDistributed(Dense(256, activation='swish'))(x[:, :, 1280:])
    
    x = Concatenate()([BatchNormalization()(cnn_p), BatchNormalization()(mp_p)])
    x = SpatialDropout1D(0.3)(x)
    
    # Deep Temporal
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
    
    x = Dense(256, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def perform_inference(model, X_test, y_test, class_mapping, num_samples=10, log_file=None):
    rev_mapping = {v: k for k, v in class_mapping.items()}
    indices = random.sample(range(len(X_test)), min(num_samples, len(X_test)))
    
    log_text = ["\n--- RANDOM INFERENCE TEST ---"]
    for idx in indices:
        sample = np.expand_dims(X_test[idx], axis=0)
        actual = rev_mapping[np.argmax(y_test[idx])]
        pred_probs = model.predict(sample, verbose=0)[0]
        pred = rev_mapping[np.argmax(pred_probs)]
        conf = np.max(pred_probs) * 100
        log_text.append(f"{'✅' if actual==pred else '❌'} | Real: {actual[:15]:<15} | Pred: {pred[:15]:<15} ({conf:.1f}%)")
        
    output = "\n".join(log_text)
    print(output)
    
    # Append to log file if specified
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(output + "\n")

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    pre_extract_full_dataset(DATASET_DIR, FEATURES_DIR)
    
    X, y, class_mapping = load_full_npy_dataset(FEATURES_DIR)
    num_classes = len(class_mapping)
    print(f"Training on {len(X)} samples across {num_classes} classes.")
    
    # Handle potentially large number of classes with class weights
    weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(enumerate(weights))
    
    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.15, random_state=SEED, stratify=y)
    
    model = build_robust_isl_model(MAX_FRAMES, COMBINED_DIM, num_classes)
    
    callbacks = [
        ModelCheckpoint("isl_full_model.h5", save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        CSVLogger("training_history.csv")  # Logs per-epoch metrics
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        batch_size=BATCH_SIZE, epochs=EPOCHS, 
                        class_weight=weight_dict, callbacks=callbacks)
    
    # Save final summary report
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    report_file = "training_summary.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== ISL MODEL TRAINING SUMMARY ===\n")
        f.write(f"Total Classes Trained: {num_classes}\n")
        f.write(f"Total Epochs Run: {len(history.history['loss'])}\n")
        f.write(f"Final Validation Accuracy: {val_acc * 100:.2f}%\n")
        f.write(f"Final Validation Loss: {val_loss:.4f}\n")
        f.write("==================================\n")
        
    print(f"\n✅ Training metrics successfully logged to 'training_history.csv' and '{report_file}'.")
    
    # Run inference and append to the report
    perform_inference(model, X_val, y_val, class_mapping, log_file=report_file)
