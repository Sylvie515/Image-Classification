# Transfer Learning for Image Classification: Scene Recognition

This project implements multi-class image classification for six scene categories using transfer learning with TensorFlow/Keras. It evaluates the performance of pre-trained models ResNet50, ResNet101, EfficientNetB0, and VGG16 on a small scene dataset, applying data augmentation and fine-tuning to improve generalization.  

Achieving 90.0% test accuracy with EfficientNetB0.

## Usage

1. Install dependencies  

   pip install tensorflow keras numpy matplotlib scikit-learn  

2. Prepare the dataset in the required folder structure  

3. Run the jupyter notebook  
   
## Dataset

The dataset consists of six scene categories: buildings, forest, glacier, mountain, sea, and street.  

Images for each class are organized into separate folders under seg_train and seg_test.  

### Folder Structure

seg_train/  
├── buildings/  
├── forest/  
├── glacier/  
├── mountain/  
├── sea/  
└── street/  

seg_test/  
├── buildings/  
├── forest/  
├── glacier/  
├── mountain/  
├── sea/  
└── street/  

Validation set: 20% stratified sampling from each class in the training set to maintain balanced class distribution.  

## Data Pre-processing

1. Resized all images to 224x224 pixels (zero-padding for aspect ratio preservation) to ensure consistent input size.  

2. Apply image augmentation to the training set to enhance generalization, including:  

   - Random cropping  

   - Random flipping  

   - Random rotation  

   - Random zoom  

   - Random contrast adjustment  

   - Random translation  

## Transfer Learning

1. Use pre-trained models ResNet50, ResNet101, EfficientNetB0, and VGG16 as base models.  

2. Model Architecture (Frozen Pre-trained Layers + Custom Classification Head)  

   The final classification layers of the pre-trained models are removed, and all convolutional base layers are frozen.  

   Use the outputs of the penultimate layer in the original pre-trained model as the features extracted from each image.

   Custom Classification Head is added. Only fine-tune and train these layers.  

   Apply early stopping: Monitors validation loss to prevent overfitting and save the best model weights.  

4. Model performance is measured on training, validation, and test sets using: Precision, Recall, AUC, and F1 score.  
