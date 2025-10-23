"""
Decode emotion predictions from raw model output
Author: Sajjad Ghaeminejad
Date: October 2025
"""

import numpy as np
import argparse

# Emotion mapping (from Leo's 3DCNN_SAMPLE_MAG_OF.ipynb)
# These are the 8 emotion classes the model was trained on
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


def decode_emotions(predictions_file, output_file=None, top_k=3):
    
    print("DECODING EMOTION PREDICTIONS")
    print("="*70)
    print()
    
    # Load predictions
    print(f"Loading predictions from: {predictions_file}")
    predictions = np.loadtxt(predictions_file, delimiter=",")
    print(f"Loaded {len(predictions)} sequences")
    print(f"Each sequence has {predictions.shape[1]} emotion scores")
    print()
    
    # Decode each sequence
    results = []
    
    print(f"Top {top_k} predictions for each sequence:")
    print("-"*70)
    
    for i, pred in enumerate(predictions):
        # Get top k emotions
        top_indices = np.argsort(pred)[::-1][:top_k]
        top_scores = pred[top_indices]
        top_labels = [EMOTION_LABELS[idx] for idx in top_indices]
        
        # Primary emotion (highest score)
        primary_emotion = top_labels[0]
        primary_confidence = top_scores[0]
        
        # Store result
        result = {
            'sequence': i,
            'primary_emotion': primary_emotion,
            'confidence': primary_confidence,
            'top_emotions': list(zip(top_labels, top_scores))
        }
        results.append(result)
        
        # Print
        if i < 10 or i % 20 == 0:  # Show first 10, then every 20th
            print(f"Sequence {i:3d}: {primary_emotion:12s} ({primary_confidence:.2%})")
            for j, (label, score) in enumerate(zip(top_labels[1:], top_scores[1:]), 2):
                print(f"             {j}. {label:12s} ({score:.2%})")
            print()
    
    # Summary statistics
    print()
    print("SUMMARY")
    print("="*70)
    
    # Count emotions
    emotion_counts = {}
    for result in results:
        emotion = result['primary_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"\nEmotion distribution across {len(predictions)} sequences:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(predictions) * 100
        print(f"  {emotion:12s}: {count:3d} sequences ({percentage:5.1f}%)")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage confidence: {avg_confidence:.2%}")
    
    # Save to file if requested
    if output_file:
        print(f"\nSaving decoded results to: {output_file}")
        with open(output_file, 'w') as f:
            f.write("sequence,primary_emotion,confidence,top2_emotion,top2_confidence,top3_emotion,top3_confidence\n")
            for r in results:
                line = f"{r['sequence']},{r['primary_emotion']},{r['confidence']:.6f}"
                for label, score in r['top_emotions'][1:]:
                    line += f",{label},{score:.6f}"
                f.write(line + "\n")
        print(f"Saved!")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode emotion predictions')
    parser.add_argument('--input', type=str, default='predictions_emotions.csv',
                        help='Input CSV file with raw predictions')
    parser.add_argument('--output', type=str, default='decoded_emotions.csv',
                        help='Output CSV file with decoded results')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    results = decode_emotions(args.input, args.output, args.top_k)
