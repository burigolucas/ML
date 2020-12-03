# Skin Lesion Classification for Melanoma Detection

This project applies a deep learning model for classification of melanoma skin lesions. 

## Dataset

The data source originates from the SIIM-ISIC 2020 competetion. In particular, the triple-stratified K-fold dataset by Chris Deotte (https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords) was used.

The code was implemented to allow using the TFRecords dataset for any image size.

## Model

The model applies transfer learning with convnets trained on the ImageNet dataset. Different convnets were considered including EfficientNets and the smaller networks MobileNet. Two different network architectures were implemented: i) a base CNN is used for feature extraction and a binary classification layer is stacked at the top to identify melanoma; ii) four CNNs are used for feature extratction of different resolution of the images. In this case, the features are concatenated and used to build the top classification layer.

## Fine tuning
The code allowes for easily turn on the fine tuning of the model. In this case, the layers corresponding to the top block of the base CNN are set to trainable during a second training step.

