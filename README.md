# Furniture-Classifier
A machine learning model is used to classify beds, sofas and chairs

## Technologies Used
  1. HTML
  2. Python
  3. Docker

## Steps Followed
1. Data Preprocessing: Loaded the dataset and performed preprocessing steps such as resizing the images, splitting the data into training and validation sets, and data augmentation. utils.py

2. Model Architecture: Selected a deep learning model architecture; Convolutional Neural Network (CNN) and defined the model architecture. model.py

3. Model Training: Trained the model on the training dataset using an optimizer and loss function. model.py, detect.py

4. Model Evaluation: Evaluated the model performance on the validation dataset and fine-tune the model if necessary. detect.py

5. Predictions: Use the trained model to make predictions on new images. flaskapp.py

## Model Prediction On Test Data
![image](https://user-images.githubusercontent.com/44322966/220816793-5bd16a80-f83b-4e43-ba71-4c554142e0cd.png)

## API 
![image](https://user-images.githubusercontent.com/44322966/220815936-e06ec83b-5698-4d38-a69e-282485a58c6d.png)

![image](https://user-images.githubusercontent.com/44322966/220816472-45993e95-61e1-4fd8-a671-601cf542c42e.png)

## CI/CD Implementation with github actions
![image](https://user-images.githubusercontent.com/44322966/220815028-62939a86-bcea-44cc-a5d1-522fbdd01de6.png)
