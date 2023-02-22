from utils import load_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import  cv2

(feature, labels) = load_data()

(x_train, x_test, y_train, y_test) = train_test_split(feature,labels, test_size=0.1)

categories = ['Bed', 'Chair', 'Sofa']

model = tf.keras.models.load_model('mymodel.h5')

#model.evaluate(x_test, y_test, verbose =1)

prediction = model.predict(x_test)

plt.figure(figsize=(9,9))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i])
    plt.xlabel('Actual:'+categories[y_test[i]]+'\n'+'Predicted:'+categories[np.argmax(prediction[i])])
    plt.xticks([])

plt.show()

# # Load the image you want to classify
# pic = cv2.imread('Paige-Sofa.jpg')

# # Preprocess the image
# pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
# pic = cv2.resize(pic, (224, 224))
# pic = np.array(pic, dtype=np.float32)

# # Normalize the image
# pic = pic / 255.0

# # Add batch dimension
# pic = np.expand_dims(pic, axis=0)

# # Make predictions on the image
# predictions = model.predict(pic)

# # Get the class with highest probability
# predicted_class = np.argmax(predictions)

# # Print the predicted class
# print('Predicted Class:', categories[predicted_class])

