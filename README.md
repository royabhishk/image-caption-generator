# Image Caption (Capstone Project - Cloudthat)

## Objective
Apply Computer Vision and Natural Language Processing to generate an English sentence that best describes an image submitted for analysis

## Approach
The images were reshaped and encoded using a pretrained CNN model (Inception V3)
The provided captions were encapsulated within start and end string qualifiers and then encoded into input and expected output sequences
These inputs were then fed into a new model that took Image Vectors and Input text Sequence and trained against expected Output Sequence

For Prediction, the Test Image was passed through encoder model again to create the Encoded Image Vector which was passed to the model along with a start qualifier to output the next words until the end of sequence qualifier is received

## Data Used
Flickr 8k Images Dataset
