# TODO



### 1. Pre-Processing [ETL Pipeline]

- [x] ffcv conversion

- [x] Extract the dataset
  - [x] Data Statistics
  - [x] Visualize data samples
- [x] Transform the dataset
  - [x] Albumentation for augmentation
- [x] Load the dataset
  - [x] Custom dataset class
  - [x] Custom dataloader class



### 2. Model Training

- [x] Model architecture creation
- [x] Model summary and parameter verification
- [x] LR finder
- [x] One Cycle Policy
- [x] Model training
- [x] Plot losses and accuracies
- [x] Plot OCP lr graph for verification
- [ ] Visualize misclassified images
- [x] Gradcam from last layer of each block
- [ ] Confusion matrix
- [x] AWS setup
- [x] AWS training
- [x] Save the model



### 3. Inference [Gradio App]

- [x] Creating space for the app
- [x] Load model
- [x] Samples
- [x] Inference with gradcam



---



# Approach

- [x] Setup workspace on Google Colab

- [x] AWS [MiniImageNet dataset]

  - [x] Train PyTorch model 
  - [x] Train Custom model

- [ ] AWS [ImageNet dataset]

  - [ ] Train custom model

    > Started training. Got around 40% accuracy in 10 epochs.
    >
    > Failed to train full model since spot instance got shutdown

