#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load the pre-trained Inception V3 model
model = InceptionV3(weights='imagenet')

# Preprocess the model
graph = tf.compat.v1.get_default_graph()


# In[2]:


get_ipython().system('pip install opencv-python')


# In[ ]:


from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

app = Flask(__name__, template_folder='C:/Users/Lenovo/Documents/KBS/templates')

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists('uploads'):
    os.mkdir('uploads')

# Set the path to the 'uploads' directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
    # Get the file from the request
    video_file = request.files['video_file']
    max_size = 50 * 1024 * 1024  # 50 MB
    if 'video_file' not in request.files:
      return jsonify({'error': 'No video file uploaded'})
    video_file = request.files['video_file']
    if video_file.content_length > max_size:
      return jsonify({'error': 'Video file size exceeds maximum allowed'})
    
    # Save the file to disk
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Split the video into frames
    frames = split_video(video_path)

    # Feed the frames into the Inception V3 model
    results = detect_objects(frames)
#C:/Users/Lenovo/Documents/KBSs/templates/
    # Display the results
    return render_template('Results.html', results=results)

def split_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate and number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an empty list to store the frames
    frames = []

    # Loop through each frame in the video
    for i in range(num_frames):
        # Read the frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if ret:
            # Resize the frame to 224x224
            resized_frame = cv2.resize(frame, (224, 224))

            # Preprocess the frame for input to the Inception V3 model
            preprocessed_frame = preprocess_input(resized_frame)

            # Add the preprocessed frame to the list of frames
            frames.append(preprocessed_frame)
        else:
            break

    # Release the video file
    cap.release()

    # Convert the list of frames to a NumPy array
    frames = np.array(frames)

    return frames

def detect_objects(frames):
    # Load the pre-trained Inception V3 model
    model = tf.keras.applications.InceptionV3(weights='imagenet')
    frames = np.resize(frames, (frames.shape[0], 299, 299, 3))

    # Make predictions on the frames
    predictions = model.predict(frames)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=3)

    # Return the results
    return decoded_predictions

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:




