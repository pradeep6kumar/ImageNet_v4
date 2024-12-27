# ResNet50 on ImageNet

## Data

- 1.28M images
- 1000 classes
- 224x224 resolution


## Steps
### Create Beton files
```bash
# For training data
$ python utils/to_beton.py \
    --data-dir /data/train \
    --write-path /data/train.beton \
    --max-resolution 256 \
    --num-workers 8

# For validation data
$ python utils/to_beton.py \
    --data-dir /data/val \
    --write-path /data/val.beton \
    --max-resolution 256 \
    --num-workers 8
```



## TODO

1. Directory structure
2. Config files usage in all files
3. AWS setup file
4. yaml file for config



### Pre-Processing [ETL Pipeline]

- [ ] ffcv conversion

- [ ] Extract the dataset
  - [ ] Data Statistics
  - [ ] Visualize data samples
- [ ] Transform the dataset
  - [ ] Albumentation for augmentation
- [ ] Load the dataset
  - [ ] Custom dataset class
  - [ ] Custom dataloader class


### Model Training
- [ ] Model architecture creation
- [ ] Model summary and parameter verification
- [ ] LR finder
- [ ] One Cycle Policy
- [ ] Model training
- [ ] Plot losses and accuracies
- [ ] Plot OCP lr graph for verification
- [ ] Visualize misclassified images
- [ ] Gradcam from last layer of each block
- [ ] Confusion matrix
- [ ] AWS setup
- [ ] AWS training
- [ ] Save the model


### Inference [Gradio App]
- [ ] Creating space for the app
- [ ] Load model
- [ ] Samples
- [ ] Inference with gradcam