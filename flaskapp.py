import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, render_template
# Load the model
model = tf.keras.models.load_model('mymodel.h5')

# Initialize the Flask application
app = Flask(__name__)




@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = imagefile.filename
    imagefile.save(image_path)
    categories = ['Bed', 'Chair', 'Sofa']
    model = tf.keras.models.load_model('mymodel.h5')
    pic = cv2.imread(image_path)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic, (224, 224))
    pic = np.array(pic, dtype=np.float32)
    pic = pic / 255.0
    pic = np.expand_dims(pic, axis=0)
    predictions = model.predict(pic)
    predicted_class = np.argmax(predictions)
    print('Predicted Class:', categories[predicted_class])
    
    return render_template('index.html', prediction= categories[predicted_class] )



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

