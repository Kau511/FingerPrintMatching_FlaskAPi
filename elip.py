from flask import Flask, Response
from config import config
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, jsonify, url_for
import pyrebase
import urllib, urllib.request
import cv2
from config import config
import io
import os
import numpy as np


firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}

application = app = Flask(__name__)
CORS(application)


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET','POST'])
def Roi():
    if 'R1' not in request.files:
        return ("uoload image to crop")
    if 'R1' in request.files:
        file = request.files['R1']
        if file and allowed_file(file.filename):
            filename1 = secure_filename(file.filename)
            file.save(filename1)
            # storage.child("finger_images/" +str(filename1)).put(file)
            # link1 = storage.child("finger_images/" + str(filename1)).get_url(None)
    # resp1 = urllib.request.urlopen(link1)
    # img1 = np.asarray(bytearray(resp1.read()), dtype="uint8")
    # image = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    image = cv2.imread(filename1,cv2.IMREAD_COLOR)
    mask = np.zeros(image.shape,dtype="uint8")
    rows, cols, _ = image.shape
    cv2.ellipse(mask, (150,170), axes=(130,175), angle=0.0, startAngle=0.0, endAngle=360.0, color=(255,255,255), thickness=-1)
    ROI = np.bitwise_and(image,mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(mask)
    result = ROI[y:y+h,x:x+w]
    mask = mask[y:y+h,x:x+w]
    result[mask==0] = (255,255,255)
    data = cv2.imencode('.png', result)[1].tobytes()
    os.remove(filename1)
    return Response(data,mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)