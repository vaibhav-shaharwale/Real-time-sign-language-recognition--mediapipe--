import numpy as np
from tensorflow import keras
model = keras.models.load_model("keypoint_classifier.h5")
import cv2
import mediapipe as mp
import numpy as np
import pickle
dict = pickle.load(open('labels.pickel', 'rb'))


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    height, width, channels = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        fin_co = []
        for i, lm in enumerate(hand_landmarks.landmark):
            #x = int(lm.x*width)
            #y = int(lm.y*height)
            fin_co.append(lm.x)
            fin_co.append(lm.y)
            #arr = np.array([x, y])
            #print(i, arr)
            #if i == 8:
             #   cv2.circle(image, (x, y), 10,  (0, 0, 255), cv2.FILLED)
        
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        pred = model.predict(np.array([fin_co]), verbose=0)
        pred = np.argmax(pred)
        prediction.append(dict[pred])
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # for grayscale
    image = cv2.putText(cv2.flip(image, 1), str(prediction),(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign language', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
     break
cap.release()