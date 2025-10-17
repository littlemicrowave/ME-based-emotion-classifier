"""
Stage 2 - Face Extraction with RetinaFace
==========================================
Extracts and aligns faces from video using RetinaFace detection
following the MEB framework methodology.

Author: Sajjad Ghaeminejad
Date: October 17, 2025
"""

import cv2
import numpy as np
import os
from retinaface import RetinaFace
from retinaface.commons.postprocess import alignment_procedure
from tqdm import tqdm

print("="*70)
print("STAGE 2: FACE EXTRACTION USING RETINAFACE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
VIDEO_FILE = 'retinaface-test-video-sajjad.mov'
OUTPUT_FOLDER = 'data/extracted_faces'
FACE_SIZE = (112, 112)  # Standard size for micro-expression datasets

# ==================================================================
# MEB-STYLE FUNCTIONS (from tools/crop_and_align.py)
# ==================================================================

def align_img(img: np.ndarray, landmarks: dict) -> np.ndarray:
    """
    Align face using eye and nose landmarks.
    Based on MEB's alignment_procedure.
    """
    result = alignment_procedure(
        img, landmarks["right_eye"], landmarks["left_eye"], landmarks["nose"]
    )
    # Handle tuple return (img, transformation_matrix)
    if isinstance(result, tuple):
        return result[0]
    return result


def crop_img(img: np.ndarray, facial_area: list) -> np.ndarray:
    """
    Crop face region with 2% extra width margin.
    Based on MEB's crop_img function.
    """
    fa = facial_area
    extra_width = int(img.shape[1] * 0.02)
    return img[fa[1]:fa[3], fa[0]-extra_width:fa[2]+extra_width]


def detect_face_and_landmarks(img: np.ndarray, threshold: float = 0.9):
    """
    Detect face and extract landmarks using RetinaFace.
    Returns landmarks and facial area coordinates.
    """
    # Ensure uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Detect face
    obj = RetinaFace.detect_faces(img, threshold=threshold)
    
    # Try lower threshold if no face found
    if not isinstance(obj, dict):
        obj = RetinaFace.detect_faces(img, threshold=0.5)
        if isinstance(obj, dict):
            threshold = 0.5
    
    if not isinstance(obj, dict):
        return None, None, threshold
    
    # Extract landmarks and align
    landmarks = obj["face_1"]["landmarks"]
    aligned_img = align_img(img, landmarks)
    
    # Detect face in aligned image
    if aligned_img.dtype != np.uint8:
        aligned_img = (aligned_img * 255).astype(np.uint8)
    
    obj_aligned = RetinaFace.detect_faces(aligned_img, threshold=threshold)
    
    if isinstance(obj_aligned, dict):
        facial_area = obj_aligned["face_1"]["facial_area"]
    else:
        facial_area = obj["face_1"]["facial_area"]
    
    return landmarks, facial_area, threshold

# ============================================================================
# MAIN PROCESSING
# =========================================================

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder: {OUTPUT_FOLDER}/\n")

# Load video
print(f"Loading video: {VIDEO_FILE}")
video = cv2.VideoCapture(VIDEO_FILE)

if not video.isOpened():
    print(f"Error: Could not open {VIDEO_FILE}")
    print("Make sure the video file exists in the current directory")
    exit(1)

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"Video is loaded successfully!")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps:.2f}")
print(f"Duration: {duration:.2f} seconds\n")

# Read all frames
print("reading video frames...")
frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Convert BGR to RGB (RetinaFace expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8)
    frames.append(frame_rgb)
video.release()

print(f"Read {len(frames)} frames\n")

# Detect face in first frame
print("detecting face in first frame using RetinaFace...")
onset = frames[0]
landmarks, facial_area, threshold = detect_face_and_landmarks(onset, threshold=0.9)

if landmarks is None:
    print("error: Could not detect face in first frame!")
    print("Suggestions:")
    print("- check that a face is clearly visible")
    print("- Try recording a new video with frontal face view")
    exit(1)

print(f"Face detected successfully!")
print(f"Detection threshold: {threshold}")
print(f"Landmarks detected: {', '.join(landmarks.keys())}\n")

# Process all frames
print("Processing all frames with RetinaFace alignment...\n")
extracted_faces = []
failed_frames = []

for i, frame in enumerate(tqdm(frames, desc="Extracting faces", unit="frame")):
    try:
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Align using landmarks from first frame
        aligned_frame = align_img(frame, landmarks)
        
        if aligned_frame.dtype != np.uint8:
            aligned_frame = (aligned_frame * 255).astype(np.uint8)
        
        # Crop face region
        cropped_face = crop_img(aligned_frame, facial_area)
        
        # Resize to standard size
        face_resized = cv2.resize(cropped_face, FACE_SIZE)
        extracted_faces.append(face_resized)
        
        # Save (convert back to BGR for OpenCV)
        filename = f"{OUTPUT_FOLDER}/frame_{i:04d}.jpg"
        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, face_bgr)
        
    except Exception as e:
        failed_frames.append((i, str(e)))

# Print results
print("\n" + "="*70)
print("EXTRACTION COMPLETE!")
print("="*70)
print(f"\nResults:")
print(f"Total frames: {len(frames)}")
print(f"Faces extracted: {len(extracted_faces)}")
print(f"Failed frames: {len(failed_frames)}")
print(f"Success rate: {len(extracted_faces)/len(frames)*100:.1f}%")
print(f"Output size: {FACE_SIZE[0]}×{FACE_SIZE[1]} pixels")
print(f"\n Saved to: {OUTPUT_FOLDER}/")

if failed_frames:
    print(f"\n Failed frames: {[f[0] for f in failed_frames[:10]]}")
    if len(failed_frames) > 10:
        print(f"   ... and {len(failed_frames)-10} more")

print("\n Method: RetinaFace + Landmark-based alignment (MEB standard)")
print("="*70)
