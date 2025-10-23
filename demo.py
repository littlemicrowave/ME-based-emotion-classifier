
"""
Stage 4: Micro-Expression Recognition Demo
Author: Sajjad Ghaeminejad
Date: October 2025

Processes video and outputs emotion predictions
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

from extract_faces_retinaface import (
    detect_face_and_landmarks, 
    align_img, 
    crop_img
)

from demo_preprocessing import (
    get_optical_flow,
    create_sequences,
    uniform_sample
)


def extract_faces_from_video(video_path, target_size=(224, 224)):

   # Extract and align faces from video using RetinaFace
    
   # Returns: Array of grayscale face frames (n_frames, 224, 224)
    print(f"Loading video: {video_path}")
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open {video_path}")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reading all frames
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    video.release()
    
    print(f"Read {len(frames)} frames ({fps:.1f} FPS)")
    
    # Detecting face in first frame
    print("Detecting face in first frame...")
    onset = frames[0]
    landmarks, facial_area, threshold = detect_face_and_landmarks(onset, threshold=0.9)
    
    if landmarks is None:
        raise ValueError("Could not detect face in first frame!")
    
    print(f"Face detected (threshold: {threshold})")
    
    # Extract and align faces from all frames
    print("Extracting faces from all frames...")
    faces = []
    failed_count = 0
    
    for i, frame in enumerate(frames):
        try:
            aligned = align_img(frame, landmarks)
            if aligned.dtype != np.uint8:
                aligned = (aligned * 255).astype(np.uint8)
            
            cropped = crop_img(aligned, facial_area)
            
            # Resizing to 224x224
            resized = cv2.resize(cropped, target_size)
            
            # Converting to grayscale (required for optical flow)
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            else:
                gray = resized
            
            faces.append(gray)
            
        except Exception as e:
            failed_count += 1
            # Using previous frame if extraction fails
            if faces:
                faces.append(faces[-1].copy())
            else:
                print(f"Error: Could not process first frame: {e}")
                raise
    
    if failed_count > 0:
        print(f"Warning: {failed_count} frames failed, used previous frame")
    
    print(f"Extracted {len(faces)} faces")
    return np.array(faces)


def preprocess_for_optical_flow(faces, step=8):

    #Preprocess faces for optical flow model
    
    #Input: (n_frames, 224, 224) grayscale faces
    #Output: (n_sequences, 7, 224, 224, 3) optical flow sequences
    
    #Args:
        #faces: Array of grayscale face frames
        #step: Step size for sequences (8=non-overlapping, 1=overlapping)

    print(f"Creating 8-frame sequences (step={step})...")
    sequences = create_sequences(faces, window_size=8, step=step)
    print(f"Created {len(sequences)} sequences")
    
    print("Extracting optical flow...")
    flow_sequences = []
    for i, seq in enumerate(sequences):
        try:
            flow = get_optical_flow(seq)
            flow_sequences.append(flow)
        except Exception as e:
            print(f"Warning: Could not process sequence {i}: {e}")
    
    result = np.array(flow_sequences)
    print(f"Optical flow extracted")
    print(f"   Shape: {result.shape}")
    print(f"   Expected: (n_sequences, 7, 224, 224, 3)")
    
    return result


def run_demo(video_path, model_path, output_csv="predictions.csv", step=8):
    #Main demo function
    
    #Args:
    #    video_path: Path to input video
     #   model_path: Path to trained model (.h5 file)
     #   output_csv: Path to save predictions
     #   step: Sequence step size (8=non-overlapping, 1=overlapping)
    
    #Returns:
     #   predictions: Raw model predictions

    print("STAGE 4: MICRO-EXPRESSION RECOGNITION DEMO")
    print("="*70)
    print()
    
    faces = extract_faces_from_video(video_path)
    preprocessed = preprocess_for_optical_flow(faces, step=step)
    
    if len(preprocessed) == 0:
        print("Error: No sequences created!")
        return None
    
    print(f"Loading model: {model_path}")
    
    def weighted_loss(y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'weighted_loss': weighted_loss},
        compile=False  # Skip compilation since we only need predictions
    )
    
    print(f"Model loaded")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
    print(f"   Data shape: {preprocessed.shape}")
    
    # 4. Run predictions
    print()
    print("Running predictions...")
    predictions = model.predict(preprocessed, verbose=1)
    
    # Handle multi-output model (emotions + AUs)
    if isinstance(predictions, list):
        emotions_pred = predictions[0]  # First output: emotions
        aus_pred = predictions[1]        # Second output: action units
        print(f"Predictions complete")
        print(f"   Emotions shape: {emotions_pred.shape}")
        print(f"   AUs shape: {aus_pred.shape}")
    else:
        emotions_pred = predictions
        aus_pred = None
        print(f"Predictions complete")
        print(f"   Shape: {emotions_pred.shape}")
    
    print()
    print(f"Saving predictions to: {output_csv}")
    
    emotions_file = output_csv.replace('.csv', '_emotions.csv')
    np.savetxt(emotions_file, emotions_pred, delimiter=",", fmt='%.6f')
    print(f"   Emotions saved to: {emotions_file}")
    
    if aus_pred is not None:
        aus_file = output_csv.replace('.csv', '_aus.csv')
        np.savetxt(aus_file, aus_pred, delimiter=",", fmt='%.6f')
        print(f"   AUs saved to: {aus_file}")
    
    print()
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"Results:")
    print(f"   Total sequences: {len(emotions_pred)}")
    print(f"   Emotion classes: {emotions_pred.shape[1]}")
    if aus_pred is not None:
        print(f"   Action units: {aus_pred.shape[1]}")
    print()
    print("Sample emotion predictions (first 3 sequences):")
    for i in range(min(3, len(emotions_pred))):
        print(f"   Sequence {i}: {emotions_pred[i]}")
    print()    
    return emotions_pred, aus_pred


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Micro-Expression Recognition Demo')
    parser.add_argument('--video', type=str, default='retinaface-test-video-sajjad.mov',
                        help='Input video path')
    parser.add_argument('--model', type=str, default='optical_flow.h5',
                        help='Model path (.h5 file)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV path')
    parser.add_argument('--step', type=int, default=8,
                        help='Sequence step size (8=non-overlapping, 1=overlapping)')
    
    args = parser.parse_args()
    
    predictions = run_demo(args.video, args.model, args.output, args.step)
