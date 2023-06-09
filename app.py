#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Load the pre-trained Inception V3 model
model = InceptionV3(weights='imagenet')

# Preprocess the model
graph = tf.compat.v1.get_default_graph()

# Freezing the weights of the model
for layer in model.layers:
    layer.trainable = False

# Last layer for feature extraction
last_layer = model.get_layer('mixed7')
print('Last layer output shape:', last_layer.output_shape)
last_output = last_layer.output

# Flattenning the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Adding a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model1 = tf.keras.models.Model(model.input, x)
model1.compile(loss='binary_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
               metrics=['acc'])

def upload_video():
    st.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection in Videos</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
            <script>
                $(document).ready(function() {
                    $('#upload_form').submit(function(event) {
                        var video_file = $('#video_file').val();
                        var search_query = $('#search_query').val();

                        if (video_file == '') {
                            alert('Please select a video file to upload');
                            event.preventDefault();
                        }

                        if (search_query == '') {
                            alert('Please enter a search query');
                            event.preventDefault();
                        }
                    });
                });
            </script>
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-3">Object Detection in Videos</h1>
                <form id="upload_form" method="post" enctype="multipart/form-data" action="{{ url_for('upload_video') }}">
                    <div class="form-group">
                        <label for="video_file">Upload a video file:</label>
                        <input type="file" class="form-control-file" id="video_file" name="video_file">
                    </div>
                    <div class="form-group">
                        <label for="search_query">Search for objects:</label>
                        <input type="text" class="form-control" id="search_query" name="search_query" placeholder="Enter a search query">
                    </div>
                    <button type="submit" class="btn btn-primary">Submit</button>
                </form>
            </div>
        </body>
        </html>
    """)
# Create the 'uploads' directory if it doesn't exist
if not os.path.exists('uploads'):
    os.mkdir('uploads')

# Set the path to the 'uploads' directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
def show_results(results):
    st.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Results</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-3">Object Detection Results</h1>
                {% if results %}
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Frame</th>
                                <th>Object</th>
                                <th>Probability</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for result in results %}
                                {% for prediction in result %}
                                    <tr>
                                        <td>{{ loop.index }}</td>
                                        <td>{{ prediction[1] }}</td>
                                        <td>{{ prediction[2]|round(2) }}</td>
                                    </tr>
                                {% endfor %}
                            {% endfor %}
                        </tbody>
                    </table>
                {% else %}
                    <p>No objects found.</p>
                {% endif %}
            </div>
        </body>
        </html>
    """, results=results)
@st.cache(allow_output_mutation=True)
def load_model():
    # Load the pre-trained Inception V3 model
    model = InceptionV3(weights='imagenet')
    return model

@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def detect_objects(frames):
    # Load the pre-trained Inception V3 model
    model = load_model()

    frames = np.resize(frames, (frames.shape[0], 299, 299, 3))

    # Make predictions on the frames
    predictions = model.predict(frames)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=3)

    # Return the results
    return decoded_predictions

def app():
    st.set_page_config(page_title='Video Object Detection App')

    st.title('Video Object Detection App')
    st.write('This app detects objects in a video using the Inception V3 model.')

    # Get the file from the user
    video_file = st.file_uploader('Upload a video', type=['mp4'])

    if video_file:
        max_size = 50 * 1024 * 1024 # 50 MB
        if video_file.size > max_size:
            st.error('Video file size exceeds maximum allowed.')
        else:
            # Save the file to disk
            video_path = os.path.join(UPLOAD_FOLDER, video_file.name)
            with open(video_path, 'wb') as f:
                f.write(video_file.getbuffer())

            # Split the video into frames
            frames = split_video(video_path)

            # Feed the frames into the Inception V3 model
            results = detect_objects(frames)

            # Display the results
            st.subheader('Results')
            for i, frame in enumerate(results):
                st.write(f'Frame {i+1}:')
                for j, prediction in enumerate(frame):
                    st.write(f'{j+1}. {prediction[1]} ({prediction[2]:.2%})')
                st.write('---')

if __name__ == '__main__':
    upload_video()
    app()
    # Load the object detection results from a file
    results = load_results_from_file("results.txt")

    # Display the results page
    show_results(results)






