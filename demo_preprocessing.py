"""
Preprocessing functions extracted from Leo's Preprocessing.ipynb
DO NOT modify parameters - they must match trained models!

Author: Sajjad Ghaeminejad
Date: October 2025
"""

import cv2
import numpy as np


def optical_strain(u, v):
    return np.sqrt(u**2 + v**2)


def normalize_img(img):
    #Normalize image to [0, 1] range
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def get_optical_flow(video):
    #Extract optical flow from grayscale video frames
    #Exact copy from Leo's Preprocessing.ipynb
    
    #Input: (n_frames, height, width) grayscale uint8
   # Output: (n_frames-1, height, width, 3) normalized flow
    
    #Important: Parameters 0.5, 4, 15, 4, 5, 1.2, 0 must NOT be changed!
    shape = (len(video) - 1, video.shape[1], video.shape[2], 3)
    uv_frames = np.zeros(shape)
    
    for i in range(len(video) - 1):
        f1 = video[i]
        f2 = video[i + 1]
        
        # Leo's exact parameters from training!
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None, 
            0.5,  
            4,    
            15,   # winsize
            4,    # iterations
            5,    
            1.2,  
            0     
        )
        
        strain = optical_strain(flow[..., 0], flow[..., 1])
        uv_frame = np.concatenate([flow, np.expand_dims(strain, 2)], 2)
        uv_frame = normalize_img(uv_frame)
        uv_frames[i] = uv_frame
    
    return uv_frames


def create_sequences(faces, window_size=8, step=1):
    #Create sliding window sequences from face frames
    
    #Input: 
     #   faces: (n_frames, height, width) array of face frames
     #   window_size: Number of frames per sequence (default 8)
     #   step: Step size between sequences (1=overlapping, 8=non-overlapping)
    
    #Output: (n_sequences, window_size, height, width) array
    sequences = []
    for i in range(0, len(faces) - window_size + 1, step):
        seq = np.array(faces[i:i+window_size])
        sequences.append(seq)
    return np.array(sequences)


def uniform_sample(frames, n_sample=8):
    #Uniformly sample n frames from a sequence
    
    #Input: Array of frames
    #Output: Uniformly sampled n frames
    indices = np.linspace(0, len(frames)-1, n_sample, dtype=int)
    return frames[indices]
