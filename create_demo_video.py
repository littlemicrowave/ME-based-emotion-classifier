
"""
Create demo video with emotion predictions overlaid
Author: Sajjad Ghaeminejad
Date: October 2025
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# Emotion labels
EMOTION_LABELS = {
    0: 'disgust',
    1: 'happiness',
    2: 'negative',
    3: 'others',
    4: 'positive',
    5: 'repression',
    6: 'sadness',
    7: 'surprise'
}

# Color for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'disgust': (0, 140, 255),      # Orange
    'happiness': (0, 255, 0),      # Green
    'negative': (0, 0, 255),       # Red
    'others': (128, 128, 128),     # Gray
    'positive': (255, 255, 0),     # Cyan
    'repression': (255, 0, 255),   # Magenta
    'sadness': (255, 0, 0),        # Blue
    'surprise': (0, 255, 255)      # Yellow
}


def create_demo_video(video_path, predictions_file, output_path, fps=30, sequence_length=8):

    #Create demo video with emotion predictions overlaid
    
    #Args:
     #   video_path: Original video file
     #   predictions_file: CSV with emotion predictions
      #  output_path: Output video file
     #   fps: Frames per second
     #   sequence_length: Number of frames per sequence (default 8)

    print("CREATING DEMO VIDEO WITH EMOTION OVERLAY")
    print("="*70)
    print()
    
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    predictions = np.loadtxt(predictions_file, delimiter=",")
    print(f"Loaded {len(predictions)} sequence predictions")
    
    # Decode predictions to emotions
    emotion_per_sequence = []
    confidence_per_sequence = []
    
    for pred in predictions:
        top_idx = np.argmax(pred)
        emotion = EMOTION_LABELS[top_idx]
        confidence = pred[top_idx]
        emotion_per_sequence.append(emotion)
        confidence_per_sequence.append(confidence)
    
    # Load video
    print(f"Loading video: {video_path}")
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {total_frames} frames")
    print(f"Creating output video: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing frames...")
    frame_idx = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        sequence_idx = frame_idx // sequence_length
        
        if sequence_idx < len(emotion_per_sequence):
            emotion = emotion_per_sequence[sequence_idx]
            confidence = confidence_per_sequence[sequence_idx]
            color = EMOTION_COLORS[emotion]
            
            # semi-transparent overlay at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            text = f"Emotion: {emotion.upper()}"
            cv2.putText(frame, text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(frame, conf_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            seq_text = f"Seq: {sequence_idx}/{len(emotion_per_sequence)-1}"
            cv2.putText(frame, seq_text, (width - 180, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        out.write(frame)
        frame_idx += 1
    
    video.release()
    out.release()
    
    print()
    print("="*70)
    print("VIDEO CREATED!")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Total frames: {frame_idx}")
    print(f"Duration: {frame_idx/fps:.1f} seconds")
    print()
    print("Emotion distribution in video:")
    emotion_counts = {}
    for emotion in emotion_per_sequence:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(emotion_per_sequence) * 100
        print(f"  {emotion:12s}: {count:3d} sequences ({percentage:5.1f}%)")


def create_emotion_clips(video_path, predictions_file, output_folder, 
                        clip_duration=3, fps=30, sequence_length=8):
    """
    Create short clips for each detected emotion
    
    Args:
        video_path: Original video file
        predictions_file: CSV with emotion predictions
        output_folder: Folder to save clips
        clip_duration: Duration of each clip in seconds
        fps: Frames per second
        sequence_length: Frames per sequence
    """
    print("="*70)
    print("CREATING EMOTION CLIPS")
    print("="*70)
    print()
    
    predictions = np.loadtxt(predictions_file, delimiter=",")
    
    emotions_list = []
    for pred in predictions:
        top_idx = np.argmax(pred)
        emotion = EMOTION_LABELS[top_idx]
        confidence = pred[top_idx]
        emotions_list.append((emotion, confidence))
    
    #  best example of each emotion
    best_examples = {}
    for seq_idx, (emotion, confidence) in enumerate(emotions_list):
        if emotion not in best_examples or confidence > best_examples[emotion][1]:
            best_examples[emotion] = (seq_idx, confidence)
    
    print(f"Found examples for {len(best_examples)} emotions:")
    for emotion, (seq_idx, conf) in sorted(best_examples.items()):
        print(f"  {emotion:12s}: Sequence {seq_idx} (confidence: {conf:.1%})")
    print()
    
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)
    
    # Load video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read all frames
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    
    print(f"Loaded {len(frames)} frames from video")
    
    # clips
    clip_frames = int(clip_duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for emotion, (seq_idx, confidence) in best_examples.items():
        # Calculate frame range
        start_frame = seq_idx * sequence_length
        end_frame = min(start_frame + clip_frames, len(frames))
        
        if end_frame - start_frame < fps:  # Skip if too short
            continue
        
        # Create clip
        output_file = f"{output_folder}/{emotion}_{confidence:.0%}.mp4"
        print(f"Creating clip: {output_file}")
        
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        for frame_idx in range(start_frame, end_frame):
            frame = frames[frame_idx].copy()
            color = EMOTION_COLORS[emotion]
            
            # Add overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Add text
            text = f"Emotion: {emotion.upper()}"
            cv2.putText(frame, text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(frame, conf_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
    
    print()
    print("="*70)
    print("EMOTION CLIPS CREATED!")
    print("="*70)
    print(f"Output folder: {output_folder}/")
    print(f"Created {len(best_examples)} clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create demo video with emotions')
    parser.add_argument('--video', type=str, default='retinaface-test-video-sajjad.mov',
                        help='Input video path')
    parser.add_argument('--predictions', type=str, default='predictions_emotions.csv',
                        help='Predictions CSV file')
    parser.add_argument('--output', type=str, default='demo_output.mp4',
                        help='Output video path')
    parser.add_argument('--clips', action='store_true',
                        help='Create individual emotion clips')
    parser.add_argument('--clips-folder', type=str, default='emotion_clips',
                        help='Folder for emotion clips')
    
    args = parser.parse_args()
    
    # Create full demo video
    create_demo_video(args.video, args.predictions, args.output)
    
    # Create emotion clips if requested
    if args.clips:
        create_emotion_clips(args.video, args.predictions, args.clips_folder)
