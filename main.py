import json
from ibm_watson import VisualRecognitionV4
from ibm_watson.visual_recognition_v4 import FileWithMetadata, AnalyzeEnums
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import cv2
from time import sleep
from playsound import playsound

from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    cam = cv2.VideoCapture(1)
    ret, frame = cam.read()
    while True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        print(faces)
        if faces != ():
            playsound('./c.mp3')
            sleep(6)
            ret, frame = cam.read()
            break
        if cv2.waitKey(1) == ord('q'):
            break

    apikey = 'aYD2W9VdJ77eqnIP6t15siH3v0Ia4yCcyb2Q8CU50ReE'
    url = 'https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/93283c62-19fa-466e-9096-0909fca098ba'
    collection = '30b9a177-3891-41e4-a4d9-64fbee24225b'

    auth = IAMAuthenticator(apikey)
    service = VisualRecognitionV4('2018-03-19', authenticator=auth)
    service.set_service_url(url)

    img_str = cv2.imencode('.jpg', frame)[1].tobytes()
    analyze_images = service.analyze(collection_ids=[collection],
                                     features=[AnalyzeEnums.Features.OBJECTS.value],
                                     images_file=[FileWithMetadata(img_str)]).get_result()


    print(analyze_images)

    if analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object'] is not None:
        if 'mask' in analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object']:
            playsound('./m.mp3')
            sleep(3)
        elif 'face' in analyze_images['images'][0]['objects']['collections'][0]['objects'][0]['object']:
            playsound('./erro.mp3')