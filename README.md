# Video Analysis with Convolutional Neural Network

### Data Prepration

Crop video frames to create training dataset. Each image belong to one class.

### Model Training

Use the dataset of above created images for model training. Different combination experementing for best model configuration analysis. Use VGG16 as backbone also.

### Video Analysis

Crop each frame of new video. Apply trained model on this frame, predict class. Put class label on the frame and merge all frames to create new labeled video.
