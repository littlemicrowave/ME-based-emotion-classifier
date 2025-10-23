
"""
Complete Micro-Expression Demo Pipeline
Runs entire pipeline: face extraction -> preprocessing -> prediction -> visualization

Author: Sajjad Ghaeminejad
Date: October 2025

Usage:
    python3 run_full_demo.py --video my_video.mov
    python3 run_full_demo.py --video my_video.mov --no-clips
    python3 run_full_demo.py --video my_video.mov --model magnification.h5
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print()
    print("="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, env=os.environ.copy())
    
    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        sys.exit(1)
    
    print(f"\nSuccess: {description} completed!")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Complete Micro-Expression Demo Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_full_demo.py --video my_video.mov
  python3 run_full_demo.py --video my_video.mov --no-clips
  python3 run_full_demo.py --video my_video.mov --step 1 --clips-only
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                        help='Input video file (required)')
    parser.add_argument('--model', type=str, default='optical_flow.h5',
                        help='Model to use (default: optical_flow.h5)')
    parser.add_argument('--step', type=int, default=8,
                        help='Sequence step size: 8=non-overlapping, 1=overlapping (default: 8)')
    parser.add_argument('--output-prefix', type=str, default='output',
                        help='Prefix for output files (default: output)')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip creating full demo video (faster)')
    parser.add_argument('--no-clips', action='store_true',
                        help='Skip creating emotion clips')
    parser.add_argument('--clips-only', action='store_true',
                        help='Only create clips, skip full video')
    parser.add_argument('--skip-prediction', action='store_true',
                        help='Skip prediction step (use existing predictions)')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Define output files
    predictions_csv = f"{args.output_prefix}_predictions.csv"
    emotions_csv = f"{args.output_prefix}_predictions_emotions.csv"
    aus_csv = f"{args.output_prefix}_predictions_aus.csv"
    decoded_csv = f"{args.output_prefix}_decoded_emotions.csv"
    demo_video = f"{args.output_prefix}_demo.mp4"
    clips_folder = f"{args.output_prefix}_clips"
    
    print()
    print("="*70)
    print("MICRO-EXPRESSION DEMO - FULL PIPELINE")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Input video: {args.video}")
    print(f"  Model: {args.model}")
    print(f"  Step size: {args.step}")
    print(f"  Output prefix: {args.output_prefix}")
    print()
    
    # Ensure TF_USE_LEGACY_KERAS is set
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    
    # STEP 1: Run prediction pipeline
    if not args.skip_prediction:
        cmd = [
            'python3', 'demo.py',
            '--video', args.video,
            '--model', args.model,
            '--output', predictions_csv,
            '--step', str(args.step)
        ]
        run_command(cmd, "Face Extraction + Preprocessing + Prediction")
    else:
        print()
        print("="*70)
        print("STEP: Skipping prediction (using existing files)")
        print("="*70)
    
    # Check if prediction files exist
    if not Path(emotions_csv).exists():
        print(f"Error: Predictions file not found: {emotions_csv}")
        print("Run without --skip-prediction to generate predictions")
        sys.exit(1)
    
    # STEP 2: Decode predictions
    cmd = [
        'python3', 'decode_predictions.py',
        '--input', emotions_csv,
        '--output', decoded_csv
    ]
    run_command(cmd, "Decode Predictions")
    
    # STEP 3: Create visualizations
    create_video = not args.no_video and not args.clips_only
    create_clips = not args.no_clips
    
    if create_video or create_clips:
        cmd = [
            'python3', 'create_demo_video.py',
            '--video', args.video,
            '--predictions', emotions_csv,
            '--output', demo_video
        ]
        
        if create_clips:
            cmd.extend(['--clips', '--clips-folder', clips_folder])
        
        description = []
        if create_video:
            description.append("Full Demo Video")
        if create_clips:
            description.append("Emotion Clips")
        
        run_command(cmd, " + ".join(description))
    
    # STEP 4: Summary
    print()
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print()
    print("Output files created:")
    print()
    
    files_to_check = [
        (emotions_csv, "Raw emotion predictions"),
        (aus_csv, "Raw AU predictions"),
        (decoded_csv, "Decoded emotion labels"),
    ]
    
    if create_video:
        files_to_check.append((demo_video, "Demo video with overlays"))
    
    for filepath, description in files_to_check:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  [{size_str:>10}] {filepath}")
            print(f"               -> {description}")
    
    if create_clips and Path(clips_folder).exists():
        clip_files = list(Path(clips_folder).glob("*.mp4"))
        print(f"  [{len(clip_files)} clips] {clips_folder}/")
        print(f"               -> Individual emotion clips")
    
    print()
    print("To view results:")
    if create_video:
        print(f"  open {demo_video}")
    if create_clips:
        print(f"  open {clips_folder}/")
    print(f"  cat {decoded_csv}")
    print()


if __name__ == "__main__":
    main()
