# Project Plan

The **goal** of the project is to develop an AI model for recognition of facial micro-expressions and emotion classification. Facial micro-expression action units (AUs) are spontaneous, subtle, and rapid muscle movements, which can be used to show the true feelings that humans may hide.

## 1. Schedule

The project will be divided into 6 stages, and we will try to follow Agile methodology throughout the implementation. Each team member will have a personal role and set of responsibilities, considering their personal skillset, to achieve maximum group efficiency.

|  Phase  |                          Description                         |        Timing       |
| :-----: | :----------------------------------------------------------: | :-----------------: |
| Stage 1 | Familiarization with toolset, dataset, and literature review | 22.09. - 28.09.2025 |
| Stage 2 |                      Data preprocessing                      |  29.09. - 5.10.2025 |
| Stage 3 |                 Model training and evaluation                |  6.10. - 12.10.2025 |
| Stage 4 |                     Demo app development                     | 13.10. - 19.10.2025 |
| Stage 5 |                   Presentation, short week                   | 20.10. - 24.10.2025 |
| Stage 6 |                            Report                            | 25.10. - 31.10.2025 |

## Project Stages

### Stage 1. Familiarization

This stage should focus on familiarization with available tools and filling theoretical gaps, with dataset acquisition and review. Team members will focus mainly on familiarization with [MEB](https://github.com/tvaranka/meb) and its dependencies (if some are unfamiliar), which provides a basis to work with the most popular micro-expression datasets and basic preprocessing tools. We should also consider other available frameworks which might streamline missing functionality of MEB.

**Stage output**

* Dataset
* Ready team members
* Start of preprocessing

### Stage 2. Preprocessing

At this stage we should be familiar with the dataset and its contents. Most focus should be put on building a robust video preprocessing pipeline, whose outputs will be used for model training as well as in the demo app development stage.

**Things to consider:**

1. Videos should be split into frames
2. Frames should be resized
3. Face bounding box detection and extraction (we can use some pre-trained model)
4. Color channels must be scaled and transformed into a single channel, for instance grayscale/chroma for efficiency
5. Frame rate downsampling
6. Applying temporal filtering, magnification, and/or optical flow extraction (we can try all and check which gives better results in modeling)
7. Some testing to verify that frames are processed properly

**Frameworks:** MEB, scikit-learn, OpenCV, NumPy, Matplotlib, and others

**Stage output**

* Preprocessed dataset (magnified frames, optical flow matrices)
* Jupyter notebook with explanations of methods and processing steps

### Stage 3. Modeling

At this stage we should come up with a model architecture; potentially we should build at least two models for AU recognition and compare their performance. We also need to consider the optimal number of input frames.

**Things to consider:**

1. Data should be split for training, validation, and testing
2. Custom training loop (must be implemented if TensorFlow is used)
3. 3D CNN
4. 2D CNN + (Bi)LSTM/GRU
5. We might also try Attention instead of LSTM/GRU
6. Since it is a classification task, loss is binary cross-entropy or categorical cross-entropy, last layer is softmax/sigmoid

**Frameworks:** TensorFlow/PyTorch, scikit-learn, Matplotlib, NumPy, Pandas, MEB, and other tools

Evaluation will be done according to F1 score, Precision, Recall, Accuracy, Confusion Matrix, and plotting of training and validation accuracy change throughout epochs.

**Stage output**

* Trained model
* Jupyter notebook with explanations, evaluation results, and visualizations

### Stage 4. Demo app

At this stage we take the winner architecture model and build an app which uses the preprocessing pipeline from [Stage 2](#stage-2-preprocessing). The app should get the input video stream (for instance webcam), preprocess input frames, feed them into the model, and render the AU predictions in real-time. This stage especially utilizes achievements of previous stages.

**Stage output**

1. Interactive demo video showing ME recognition in real-time
2. App script

### Stage 5. Presentation

Preparation of presentation materials and a video of the demo app, and other potential tasks.

**Stage output**

* Presentation

### Stage 6. Report

Preparation of report in LaTeX format. GitHub CI pipeline for building the PDF document is ready at the moment of writing this project plan. The report should be based on the Jupyter notebooks from stages [Stage 2](#stage-2-preprocessing), and [Stage 3](#stage-3-modeling).

**Stage output**

* PDF

Everything else in the plan remains intact.

## Preliminary roles
| Project member     |    Role    |   Main Task  |
|:------------------:|:----------:|:------------:|
| Leo Davidov        |            |              |
| Raffaele Sali      |            |              |
| Sajjad Ghaeminejad |            |              |
| Zhou Yang          |            |              |
| Anatolii Fedorov   |            |              |
| Timofei Polishchuk |            |              |

## Project follow-up

|  Phase  |                          Description                         |  Stage Achievement  |
| :-----: | :----------------------------------------------------------: | :-----------------: |
| Stage 1 | Familiarization with toolset, dataset, and literature review |  |
| Stage 2 |                      Data preprocessing                      |  |
| Stage 3 |                 Model training and evaluation                |  |
| Stage 4 |                     Demo app development                     |  |
| Stage 5 |                   Presentation, short week                   |  |
| Stage 6 |                            Report                            |  |









