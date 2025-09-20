# Project plan
The **goal** of the project is to develep AI model for recognition of facial micro expressions and emotion classification. Facial micro expression action-units (AUs) are spontanous, subtle, and rapid musle movements, which can be used to show the true feelings that human may hide.   
## 1. Schedule
Project will be devided into 5 stages, and we will try to follow Agile methodology throughout the implementation. Each team member will have personal role and set of responsibilities, considering personal skillset, to achieve maximum group efficiency.
| Phase    | Description                                                   | Timing              |
|:--------:|:-------------------------------------------------------------:|:-------------------:|
| Stage 1  |  Familirization with toolset, dataset, and literature review  | 22.09. - 28.09.2025 |
| Stage 2  |  Data preprocessing                                           | 29.09. - 5.10.2025  |
| Stage 3  |  Model training and evaluation                                | 6.10. - 12.10.2025  |
| Stage 4  |  Demo app development                                         | 13.10. - 19.10.2025 |
| Stage 5  |  Presentation, short week                                     | 20.10. - 24.10.2025 |
| Stage 6  |  Report                                                       | 25.10. - 31.10.2025 |
## Project stages

### Stage 1. Familirization
This stage should focus on familirization with available tools, and filling theoretical gaps, with dataset aquisition and review. As the reference team members put main focus on familirization with [MEB](https://github.com/tvaranka/meb), and its dependecies (if some are unfamiliar), which provides a basis to work with the most popular micro expression datasets, and basic preprocessing tools. We also should consider other available frameworks which might streamline the missing functionality of MEB.  

**Stage output**  
1. Dataset
2. Ready team memebers

### Stage 2. Preprocessing
At this stage we should be familiar with dataset, and it's contents. Most focus should be put on building robust video preprocessing pipeline, which outputs will be used for model training stage as well as in the demo app development stage.  
**Things to consider:** 
1.  Videos should be splitted into frames
2.  Frames should be resized
3.  Color channels must be scaled and tranformed into a single-channel for instance grayscale/chromo for efficiency
4.  Face bounding box detection and extraction, here we can use some pre-trained model
5.  Applying magnification and/or optical flow extraction (we can try all three and see which gives better result in modelling)
6.  Some testing to verify that frames are processed properly  

Frameworks: MEB, scikit-learn, OpenCV, numpy, matplotlib and others  

**Stage output**

1. Preprocessed dataset (frames, optical flow, magnified videos)
2. Jupyter notebook with explanations what is going on, and methods.

### Stage 3. Modeling
At this stage we should come up with model architecture, potentially we should build at least two models for AU recognition, and compare the performance. We also need to consider optimal number of input frames.
**Things to consider:**
0. Data should be splitted for training + validation + testing
1. Custom training loop (must be if TensorFlow is used)
2. 3DCNN
3. 2DCNN + (bi)LSTM/GRU
4. We also might try Attention instead of LSTM/GRU
5. Since it is a classifiction task -> loss is binary crossentropy or categorical cross entropy, last layer is softmax/sigmoid  

Frameworks: TensorFlow/PyTorch, scikit, matplotlib, numpy, pandas, MEB, and other tools  

Evalutation will be done according to F1 score, Precision, Recall, Accuracy, Confusion Matrix, plotting of training validation accuracy change throughout training epochs.  

**Stage output**

1. Trained model
2. Jupyter notebook with explanations what is going on, evaluation results and visualizations

### Stage 4. Demo app.
At this stage we take out winner architecture model and build an app which uses preprocessing pipeline from **Stage 2**. App should get the input video stream (for instance webcam) preprocesses input frames and feed it into the model, and render the AU-predictions in real-time. This stage, especially, utilizes achievements of previous stages.  

**Stage output**
1. Interactive demo video showing ME recognition real-time
2. App script

### Stage 5. Presentation
Preparation of presenation materials and video of Demo app, and other potential tasks.

**Stage output**
 * Presentation

### Stage 6. Report
Preparation of report in LaTex format. GitHub CI pipeline for building PDF document is ready at the moment of writing this project plan.

**Stage output**
 * PDF










