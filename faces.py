import numpy as np
import cv2
import time
import os
import os.path
import h5py
import tensorflow as tf
import pickle
import pandas as pd

from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import load_model


K.set_image_data_format('channels_first')
np.set_printoptions(threshold=np.nan)


# Generates a vector encoding of the paramater image of length 128
# Paramaters:   image_path      path to a target image for encoding
#               model           model used to generate encoding from image 
# Return:       numpy array     encoding of the image
def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


# Loads, resizes and saves an image
# Paramaters:   image_path      path to a target image
# Return:       Void
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(image_path, img)


# checks whether the input face is a registered user or not
# returns boolean dependant on registration, the identity if registered, and the distance between vectors
# Parameters:   image_path      Path to a image used for recognition
#               database        Database of registered users
#               threshold       Accuracy paramater
# Returns:      Float           distance between image encoding and closest registered encoding 
#               String          name of the match if one exists
#               Boolean         was image of a registered user
def find_face_realtime(image_path, database, model, threshold):
    encoding = img_to_encoding(image_path, model) 
    registered = False
    min_dist = 99999
    identity = 'Unknown Person'

    # loop over all the recorded encodings in database
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding))

        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        registered = False
    else:
        registered = True

    return min_dist, identity, registered


# Detects face within a frame over a period of time
# Paramaters:   None
# Returns:      Boolean     was there a face in the frame
#def detect_face(database, model):
def detect_face():
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  # WIDTH
    capture_obj.set(4, 480)  # HEIGHT
    face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    face_found = False
    dirname = os.path.dirname(save_loc)

    # If saved image directory does not exist create one
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # run the webcam for given seconds
    req_sec = 3
    loop_start = time.time()
    elapsed = 0

    while(elapsed < req_sec):
        curr_time = time.time()
        elapsed = curr_time - loop_start
        ret, frame = capture_obj.read()     # Frame by frame capture
        frame = cv2.flip(frame, 1, 0)       # Mirror the frame, why?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to gray scale for recogition
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Detect face if it exists

        for (x, y, w, h) in faces:
            roi_color = frame[y-90:y+h+70, x-50:x+w+50] # Region of interest (suspected face)
            cv2.imwrite(save_loc, roi_color)            # Save detected face
            cv2.rectangle(frame, (x-10, y-70),(x+w+20, y+h+40), (15, 175, 61), 4)   # Define the bounding box for the face

        cv2.imshow('frame', frame)  # Display the color frame with the bounding box

        # Close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()

    img = cv2.imread(save_loc)
    if img is not None:
        face_found = True
    else:
        face_found = False

    return face_found


# Takes a video stream and does facial recognition on any individuals in the frame by comparing
# vector embeddings against an existing database
# Paramaters:   database        database of registered users
#               model           siamese model used for verification of registration
#               threshold       accuracy paramater
# Returns:      void
def detect_face_realtime(database, model, threshold=0.7):
    text = ''
    font = cv2.FONT_HERSHEY_SIMPLEX
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  # WIDTH
    capture_obj.set(4, 480)  # HEIGHT
    face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    dirname = os.path.dirname(save_loc)

    # If no database exists create one
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print('**************** Enter "q" to quit **********************')
    prev_time = time.time()
    while(True):
        ret, frame = capture_obj.read() # Capture images frame by frame
        frame = cv2.flip(frame, 1, 0)   # Mirror the image, Why?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to gray scale for detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Detect face

        # Display the resulting frame
        for (x, y, w, h) in faces:
            roi_color = frame[y-90:y+h+70, x-50:x+w+50] # Define a region for face
            cv2.imwrite(save_loc, roi_color)    # Save the detected region of interest
            curr_time = time.time()     # Keeps track of waiting time for face recognition

            if curr_time - prev_time >= 3:
                img = cv2.imread(save_loc)
                if img is not None:
                    resize_img(save_loc)
                    min_dist, identity, registered = find_face_realtime(save_loc, database, model, threshold)

                    if min_dist <= threshold and registered:
                        text = 'User: ' + identity  # Set text for display on frame
                    else:
                        text = 'Unkown user'    # Set text for display on frame

                    print('distance:' + str(min_dist))
                prev_time = time.time() # Save the time when the last face recognition task was done

            cv2.rectangle(frame, (x-10, y-70), (x+w+20, y+h+40), (15, 175, 61), 4)  # Draw a rectangle bounding the face
            cv2.putText(frame, text, (50, 50), font, 1.8, (158, 11, 40), 3)     # Put text on frame

        cv2.imshow('frame', frame)  # Display the frame with bounding rectangle and text

        # close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()
    img = cv2.imread(save_loc)


# The model uses **Triplet loss function**.
# Paramaters:   y_true      do i need this?
#               y_pred      list containing three objects:
#                           anchor(None, 128), encoding for the anchor image
#                           positive(None, 128), encoding for the positive image
#                           negative(None, 128), encoding for the negative image
#               alpha       parameter to prevent convergence to zero
# Returns:      Float       loss
def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor = y_pred[0] 
    positive = y_pred[1]
    negative = y_pred[2]
    
    # triplet formula components
    positive_dist = tf.reduce_sum( tf.square(tf.subtract(anchor, positive)) )
    negative_dist = tf.reduce_sum( tf.square(tf.subtract(anchor, negative)) )
    basic_loss = positive_dist - negative_dist + alpha

    return tf.maximum(basic_loss, 0.0)


# Loads the pre-trained model
# Parameters:   none
# Returns:      void
def load_FRmodel():
    model = load_model('model.h5', custom_objects={'triplet_loss': triplet_loss})
    return model


# Creates and initializes or loads a database of registered users. Dictionary maps each registered user with their face encoding.
# Parameters:   none
# Returns:      dictionary  database of loaded users or empty dictionary
def ini_user_database():
    user_db = {}

    # check for existing database
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)

    return user_db


# Adds a new user using image taken from webcam
# Paramaters:   user_db     database of registered users
#               model       model used for the encoding of image to vector
# Returns:      void
def add_user(user_db, model, name):
    face_found = detect_face(user_db, model)

    if face_found:
        resize_img("saved_image/1.jpg")
        #if name not in user_db:
        add_user_img_path(user_db, model, name, "saved_image/1.jpg")
        #else:
        #    print('The name is already registered! Try a different name.........')
    else:
        print('There was no face found in the visible frame. Try again...........')
        

# Adds a new user face to the database using their image stored on disk using the image path
# May deperacate and absorb into the add_user method
# Paramaters:   user_db     database of registered users
#               model       model used for the encoding of image to vector
#               name        string with name of person in the image
#               img_path    image path for image to be added to database
# Returns:      void
def add_user_img_path(user_db, model, name, img_path):
    if name not in user_db:
        user_db[name] = img_to_encoding(img_path, model)  
        filename = 'database/user_dict.pickle'  
        dirname = os.path.dirname(filename)
        
        # If no database exists create one
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Save the database
        with open('database/user_dict.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('User ' + name + ' added successfully')
    else:
    #Figure out how to change one to one mapping to one to list
        print('Picture added to existing user.')
        

# Deletes a registered user from database
# Paramaters:   user_db     database of registered users
#               name        name of the user to be removed
# Returns:      Boolean     was user removed
def delete_user(user_db, name):
    popped = user_db.pop(name, None)

    if popped is not None:
        with open('database/user_dict.pickle', 'wb') as handle:
            pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    else:
        return False


# For making this face recognition system we are going to take the input image, find its encoding and then 
# see if there is any similar encoding in the database. We define a threshold value to decide whether the 
# images are similar based on the similarity of their encodings.
def find_face(image_path, database, model, threshold=0.6):
    encoding = img_to_encoding(image_path, model)
    min_dist = 99999

    # loop over all the recorded encodings in database
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        identity = 'Unknown Person'
    else:
        print(str(identity) + ", L2 distance: " + str(min_dist))

    return min_dist, identity


# for doing face recognition 
def face_recognition(user_db, FRmodel, threshold=0.7, save_loc="saved_image/1.jpg"):
    # we can use the webcam to capture the user image then get it recognized
    face_found = detect_face(user_db, FRmodel)

    if face_found:
        resize_img("saved_image/1.jpg")
        find_face("saved_image/1.jpg", user_db, FRmodel, threshold)
    else:
        print('There was no face found in the visible frame. Try again...........')


if __name__ == '__main__':
    model = load_FRmodel()
    print('\n\nModel loaded...')

    user_db = ini_user_database()
    print('User database loaded')
    
    choice = 'y'
    while(choice == 'y' or choice == 'Y'):
        user_option = input('\nEnter choice \n1. Face Recognition\n2. Add or Delete user\n3. Quit\n')

        # User selected facial recognition
        if user_option == '1':
            os.system('cls' if os.name == 'nt' else 'clear')
            detect_face_realtime(user_db, model, threshold=0.6)

        # User selected add or delete user from db
        elif user_option == '2':
            os.system('cls' if os.name == 'nt' else 'clear')
            print(
                '1. Add user using saved image path\n2. Add user using Webcam\n3. Delete user\n')

            option = input()
            name = input('Enter the name of the person\n')

            # User opted to add image from path
            if option == '1':
                img_path = input(
                    'Enter the image name with extension stored in images/\n')
                add_user_img_path(user_db, model, name, 'images/' + img_path)

            # User opted to use webcam to add user
            elif option == '2':
                add_user(user_db, model, name)

            # User opted to delete user from the db
            elif option == '3':
                delete_user(user_db, name)
            else:
                print('Invalid choice....\n')

        # User selected quit
        elif user_option == '3':
            break

        else:
            print('Invalid choice....\nTry again?\n')

        choice = input('Continue? y or n\n')
        os.system('cls' if os.name == 'nt' else 'clear')    # clear the screen
