import cv2
from keras.models import model_from_json
import numpy as np
#from keras_preprocessing.image import load_img
json_file = open("EffNet93.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("EffNet93.h5")
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    # feature = np.array(image)
    feature = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature = feature.reshape(1,224,224,3)
    return feature # /255.0
img_width = 224
img_height = 224
def preprocess_live_frame(frame):
    # Resize the frame to match the input size of the trained model
    resized_frame = cv2.resize(frame, (img_width, img_height))
    frame_normalized = resized_frame / 255.0  # Normalize pixel values
    return np.expand_dims(resized_frame, axis=0)

webcam=cv2.VideoCapture(2) # sesuaikan camera 
labels = {0:'Surprise', 1:'Angry', 2:'Neutral', 3:'Sad', 4:'Happy', 5:'JagoanNeon'}

while True:
    i,im=webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try: 
        for (p,q,r,s) in faces:
            image = rgb[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(0,255,0),2)
            image = cv2.resize(image,(224,224))
#             img = preprocess_live_frame(image)
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            # print("Predicted Output:", prediction_label)
            # cv2.putText(im,prediction_label)
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        cv2.imshow("Alreza",im)
        cv2.waitKey(27)
    except cv2.error:
        pass