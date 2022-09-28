from crypt import methods
import cv2

import img_proc
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np
from io import BytesIO

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from pathlib import Path
import io


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

db = SQLAlchemy(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///paints_db"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
migrate = Migrate(app, db)


class Mask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    parentId = db.Column(db.Integer, nullable=False)
    mask = db.Column(db.Text, nullable=False)
    polygon = db.Column(db.Text, nullable=False)
    coords = db.Column(db.Text, nullable=False)


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route("/", methods=["GET"])
def status_check():
    return jsonify(status="OK")


@app.route("/predict", methods=['POST'])
@cross_origin()
def change_color():
    uploaded = request.form['uploaded']
    file1 = request.files['org_image']
    org_img = Image.open(file1.stream)

    if uploaded == 'true':
        file2 = request.files['new_image'].stream
    else:
        file2 = Mask.query.filter_by(id=request.form['id']).first().mask
        file2 = io.BytesIO(file2)

    new_img = Image.open(file2)

    r = int(request.form['r'])
    g = int(request.form['g'])
    b = int(request.form['b'])
    x = int(request.form['x'])
    y = int(request.form['y'])
    width = int(request.form['width'])
    height = int(request.form['height'])
    color = [r, g, b]
    position = (x, y)
    if uploaded == "true":
        org_img = np.array(org_img)
        org_img = cv2.resize(org_img, (width, height))
        new_img = np.array(new_img)
        new_img = cv2.resize(new_img, (width, height))
    else:
        org_img = np.array(org_img)
        new_img = np.array(new_img)
    final_img = img_proc.changeColor(
        org_img=org_img, new_img=new_img, position=position, color=color, uploaded=uploaded == "true")
    final_img = Image.fromarray(final_img)

    return serve_pil_image(final_img)


@app.route("/masks", methods=['GET'])
@cross_origin()
def get_masks():
    id = request.args.get("id")
    masks = Mask.query.filter_by(parentId=id).all()
    result = []
    for item in masks:
        del item.mask
        del item._sa_instance_state
        result.append(item.__dict__)

    return jsonify(result)


if __name__ == '__main__':
    print('Application Started .......')
    app.run()
