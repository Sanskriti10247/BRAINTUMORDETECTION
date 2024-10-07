**Demo Video : [https://www.loom.com/share/e335b45a996842968ac9a16a758e1d0e?sid=e3743ccf-85a1-46df-accd-83b65140528b](url)**

**Documentation**

1. Preprocessing Steps
Dataset Description:
The dataset consists of MRI brain scan images, structured into two main folders:

train: Contains MRI images used for training, alongside their respective label files.
valid: Contains MRI images for validation, also paired with their corresponding labels.
Each of these folders is organized into subfolders:

images: This contains the actual MRI scans.
labels: Each image has a corresponding annotation file in YOLO format (.txt), which includes the bounding box and class label.
Class Labels:
positive: MRI images indicating the presence of a brain tumor.
negative: MRI images that do not show any tumor.
Preprocessing Steps:
Image Resizing: All images are resized to 640x640 pixels, a common input size for YOLO models to ensure uniformity and reduce computational overhead.
Normalization: The pixel intensity values were scaled to a range between [0, 1] by dividing by 255 to standardize inputs.
Augmentation (optional): Data augmentation techniques such as horizontal/vertical flipping, random rotations, and scaling were applied to introduce variability in the training data, enhancing the model's ability to generalize.
Annotation Format: Each image has a corresponding .txt file with:
The class label (0 for 'negative' and 1 for 'positive').
The bounding box coordinates (normalized to image size).
2. Model Architecture
Pre-trained Model Used: YOLOv8
We used the YOLOv8 model, particularly its lightweight version (YOLOv8s), which strikes a balance between accuracy and speed, making it ideal for real-time applications.
Modifications:
The output layer was modified to perform binary classification for our task, detecting whether the image contains a tumor or not.
Why YOLOv8?
Speed: YOLOv8 is known for its high inference speed, making it suitable for time-sensitive tasks.
Accuracy: The model is highly accurate and can be fine-tuned for domain-specific tasks like medical image analysis.
3. Training Process
Hyperparameters:
Batch Size: 16
Image Size: 640x640 pixels
Epochs: 30
Learning Rate: 0.01
Optimizer: Stochastic Gradient Descent (SGD) with momentum (0.937) for faster convergence.
Loss Functions: Standard YOLO loss functions, which include:
Bounding box regression for precise tumor localization.
Classification to distinguish between tumor and no-tumor.
Objectness to assess whether an object (tumor) exists in a particular region.
Data Augmentation:
Random Flipping: Images were randomly flipped horizontally and vertically to introduce spatial variability.
Rotations: Random rotations were applied to make the model robust to different orientations.
4. Evaluation Metrics
The model was evaluated using several metrics to gauge its performance on the validation set:

Accuracy:
Measures the proportion of correctly classified images (tumor vs. no tumor).
Precision:
Precision
=
True Positives
True Positives
+
False Positives
Precision= 
True Positives+False Positives
True Positives
​
 
Reflects the model's ability to correctly detect tumors without false positives.
Recall:
Recall
=
True Positives
True Positives
+
False Negatives
Recall= 
True Positives+False Negatives
True Positives
​
 
Measures how well the model detects actual tumors, even if they are small or hard to see.
F1-Score:
F1-Score
=
2
×
Precision
×
Recall
Precision
+
Recall
F1-Score=2× 
Precision+Recall
Precision×Recall
​
 
The harmonic mean of precision and recall, balancing the two to provide a more comprehensive metric.
Observations:
High Recall: The model consistently detected tumors, showing sensitivity in catching most tumor cases.
Lower Precision: A relatively higher number of false positives suggested that the model was sometimes overly cautious, marking non-tumor images as tumors.
5. Instructions for Running the Code
1. Clone the YOLOv8 Repository:
Download the YOLOv8 code from the official repository or your custom GitHub project.
2. Prepare the Dataset:
Organize your dataset into the train and valid folders with images and labels subfolders as per YOLO's required format.
Ensure the data.yaml file is configured with the correct paths to your dataset:
train: /path/to/train/images
val: /path/to/valid/images
names: ['negative', 'positive']
3. Train the Model:
Run the following command to start training:

4. Evaluate the Model:
Once training is completed, evaluate the model using the validation set:
5. Test the Model:
To test the model on new images:
This command will use the trained model to classify new MRI images and produce predictions.
6. Conclusion
The YOLOv8 model was fine-tuned to detect brain tumors from MRI images. The model achieved high recall, making it highly sensitive to tumor detection, which is crucial in medical applications. However, the precision could be improved to reduce the number of false positives. Further tuning, possibly with a more balanced dataset or additional training techniques, could improve its performance, particularly by reducing over-prediction of tumors. Overall, the model holds promise for real-world medical applications where sensitivity is key.










