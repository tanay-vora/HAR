import os
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from collections import Counter

# Define the Flask app
app = Flask(__name__)

# Load your model (make sure the path is correct)
model = tf.keras.models.load_model(r'C:\Users\Tanya\Desktop\deployement_flask\HAR.hdf5')

# Define a mapping of integer predictions to action labels
action_labels = {
    1: 'Clapping',
    2: 'Meet and Split',
    3: 'Sitting',
    4: 'Standing Still',
    5: 'Walking',
    6: 'Walking While Reading Book',
    7: 'Walking While Using Phone'
}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 10:  # Collect 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and normalize each frame
        processed_frame = cv2.resize(frame, (256, 256))  # Resize to 256x256
        processed_frame = processed_frame / 255.0  # Normalize
        frames.append(processed_frame)

    cap.release()
    
    # If there are fewer than 10 frames, pad with the last frame
    while len(frames) < 10:
        frames.append(frames[-1])  # Pad with the last frame

    # Convert frames to numpy array and reshape to (1, 10, 256, 256, 3)
    input_data = np.array(frames).reshape(1, 10, 256, 256, 3)

    # Make the prediction
    predictions = model.predict(input_data)  # Shape should be (1, num_classes)

    # Get the index of the action class with the highest probability
    action_index = np.argmax(predictions[0])  # Take the first element (batch size = 1)

    return action_index  # Return the index of the action

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'videofile' not in request.files:
            return render_template('index.html', prediction="No file part")
        
        file = request.files['videofile']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")
        
        if file:
            # Save the uploaded video
            file_path = f"./videos/{file.filename}"
            file.save(file_path)

            # Process the video and get the predicted action integer
            action_int = process_video(file_path)

            # Clean up the saved video file if desired
            os.remove(file_path)

            # Map the predicted action integer to a string label
            action_label = action_labels.get(action_int, "Unknown Action")

            # Render the template with the prediction result
            return render_template('index5.html', prediction=action_label)

    return render_template('index5.html')

if __name__ == '__main__':
    app.run(debug=True)
