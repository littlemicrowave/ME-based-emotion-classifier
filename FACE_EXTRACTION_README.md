# Stage 2: Face Extraction with RetinaFace

**Project:** ME-based Emotion Classifier  
**Stage:** 2 - Data Preprocessing  
**Author:** Sajjad Ghaeminejad  
**Date:** October 17, 2025

## Overview

This stage extracts and aligns faces from video frames using RetinaFace detection, following the MEB (Micro-Expression Benchmark) framework methodology.

## Method

### 1. Face Detection
- **Tool:** RetinaFace
- **Package:** `retina-face`
- **Function:** `RetinaFace.detect_faces()`
- **Threshold:** 0.9 (fallback to 0.5 if needed)

### 2. Face Alignment
- **Function:** `alignment_procedure()` from `retinaface.commons.postprocess`
- **Landmarks:** Right eye, left eye, nose
- **Purpose:** Standardize face orientation across all frames

### 3. Processing Pipeline
```
Video Input
    ↓
Frame Extraction (BGR → RGB)
    ↓
RetinaFace Detection (first frame)
    ↓
Extract Facial Landmarks
    ↓
For Each Frame:
    ├→ Align using landmarks
    ├→ Crop face region (+2% margin)
    ├→ Resize to 112×112
    └→ Save as JPEG
```

## Results

- **Input:** `retinaface-test-video-sajjad.mov`
- **Output:** `data/extracted_faces/`
- **Face size:** 112×112 pixels (RGB)
- **Format:** JPEG
- **Total faces:** 1,333
- **Success rate:** 100%

## Files Structure
```
ME-based-emotion-classifier/
├── extract_faces_retinaface.py    # Main extraction script
├── retinaface-test-video-sajjad.mov  # Input video
├── requirements.txt                # Python dependencies
├── data/
│   └── extracted_faces/            # Output faces
│       ├── frame_0000.jpg
│       ├── frame_0001.jpg
│       └── ... (1,333 files)
├── FACE_EXTRACTION_README.md       # This file
└── venv/                           # Virtual environment (not in git)
```

## Dependencies

See `requirements.txt`:
```
opencv-python>=4.12.0
numpy>=2.2.0
pillow>=12.0.0
matplotlib>=3.10.0
retina-face>=0.0.17
tf-keras>=2.20.0
jupyter>=1.0.0
nbformat>=5.0.0
tqdm>=4.67.0
```

## Compatibility with MEB Datasets

This preprocessing method matches the approach used by existing micro-expression datasets in MEB:

| Aspect | Our Method | MEB Datasets |
|--------|------------|--------------|
| Detector | RetinaFace | RetinaFace |
| Alignment | `alignment_procedure()` | `alignment_procedure()` |
| Landmarks | Eyes + Nose | Eyes + Nose |
| Output Size | 112×112 | 112×112 |
| Source | `meb/tools/crop_and_align.py` | Same |

## Usage

### Setup
```bash
# Clone repository
git clone https://github.com/littlemicrowave/ME-based-emotion-classifier.git
cd ME-based-emotion-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Extraction
```bash
python3 extract_faces_retinaface.py
```

## Next Steps

- **Stage 3:** Model training and evaluation (winner model selection)
- **Stage 4:** Integration with demo application

## References

1. MEB Framework: https://github.com/tvaranka/meb
2. MEB Tools: `tools/crop_and_align.py`
3. RetinaFace: Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). "Retinaface: Single-shot multi-level face localisation in the wild." CVPR 2020.

## Author
Face extraction implemented by Sajjad Ghaeminejad
