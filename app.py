import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session
from PIL import Image
import face_recognition as frg
from werkzeug.utils import secure_filename
import cv2
from load import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tkinter as tk


global model

model = init()

# create face recognition model by removing last layer
face_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# add new output layer for face recognition
output = Dense(1, activation='sigmoid')(face_model.output)
face_recognition_model = Model(inputs=face_model.input, outputs=output)

# compile model
face_recognition_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

app = Flask(__name__)
app.secret_key = 'secret'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
# Set a global variable in app.config
app.config['GLOBAL_VAR'] = None

# Check whether the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process the upload photo
def preprocess_image(img):
    # Convert the image to RGB if it has only one channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Resize the image to 250*250
    resized = cv2.resize(img, (250, 250))
    # Normalize the image to have values between 0 and 1
    normalized = resized / 255.0
    # Reshape the image to a 4D tensor (batch size, height, width, channels)
    reshaped = np.reshape(normalized, (1, 250, 250, 3))
    return reshaped

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/scan")
def scan():
    cascPath = r"haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    cv2.startWindowThread()
    cam = cv2.VideoCapture(0)
    for i in range(300):
        ret, frame = cam.read()

        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.namedWindow("Face recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face recognition", 800, 600)
        cv2.moveWindow("Face recognition", 380, 110)
        cv2.imshow("Face recognition", frame)
        cv2.setWindowProperty("Face recognition", cv2.WND_PROP_TOPMOST, 1)

        if len(faces) > 0 and i > 50:
            os.chdir(app.config['UPLOAD_FOLDER'])
            cv2.imwrite("scannedPhoto.jpg", frame)
            os.chdir("../")
            break

        k = cv2.waitKey(1)
        if k % 256 == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    return ('', 204)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if 'photo' in request.files:
            photo = request.files['photo']

            # Check if the file is an allowed file type
            if not allowed_file(photo.filename):
                flash('Invalid file type. Only PNG, JPG, JPEG, and GIF files are allowed.', 'error')
                return redirect(url_for('login'))

            # Save the file to the uploads folder
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # load the image and prepare it for classification
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            preprocessed_img = preprocess_image(img)

        # Retrieve the scanned photo
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "scannedPhoto.jpg")
            img = cv2.imread(filepath)
            preprocessed_img = preprocess_image(img)
            # Remove the photo from the working directory
            os.remove(filepath)

        # Use the pre-trained model to predict
        prediction_masked = face_recognition_model.predict(preprocessed_img)

        # Retrieve the global var
        prediction_unmasked = app.config['GLOBAL_VAR']
        # compute cosine similarity
        similarity = np.dot(prediction_unmasked, prediction_masked.T) / (
                np.linalg.norm(prediction_unmasked) * np.linalg.norm(prediction_masked))

        # check if similarity score is below threshold
        if similarity < 0.85:
            flash('Sorry, we could not recognize your face with mask. Please try again.', 'error')
            return redirect(url_for('login'))

        flash('Login successful!', 'success')
        return redirect(url_for('profile', username=username))

    # If the request method is GET, show the Login Page
    return render_template('login.html')

# Route for the Signup Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get the username and photo from the form
        username = request.form['username']
        if 'photo' in request.files:
            photo = request.files['photo']

            # Check if the file is an allowed file type
            if not allowed_file(photo.filename):
                flash('Invalid file type. Only PNG, JPG, JPEG, and GIF files are allowed.', 'error')
                return redirect(url_for('signup'))

            # Save the file to the uploads folder
            filename = secure_filename(photo.filename)
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # load the image and prepare it for classification
            img_unmasked = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            preprocessed_img_unmasked = preprocess_image(img_unmasked)

        # Retrieve the scanned photo
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "scannedPhoto.jpg")
            img = cv2.imread(filepath)
            preprocessed_img_unmasked = preprocess_image(img)
            # Remove the photo from the working directory
            os.remove(filepath)

        # Use the pre-trained model to predict
        prediction = face_recognition_model.predict(preprocessed_img_unmasked)
        app.config['GLOBAL_VAR'] = prediction

        # Redirect to the Login Page with a success message
        flash('Signup successful!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

# Profile page
@app.route('/profile/<username>')
def profile(username):
    # if 'username' in session:
    #     return render_template('profile.html', username=username)
    # else:
    #     return redirect(url_for('login'))
    return render_template('profile.html', username=username)

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)