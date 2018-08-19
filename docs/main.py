import io
import os
import shutil

from PIL import Image
from flask import Flask, redirect, request, jsonify
from keras import models
import subprocess

app = Flask(__name__)
model = None

model_url = "https://www.dropbox.com/s/4lis0cjju5ounlg/dual_model.tar"
model_fn = model_url.split("/")[-1]


# def load_model():
#     global model
#     URL = 'https://www.dropbox.com/s/qse0we8dpv4jhhm/my_model.h5'
#     filepath = URL.split("/")[-1]
#     if not os.path.exists(filepath):
#         os.system("wget " + URL)
#         print("saved to " + filepath)
#     # sleep(5)
#
#     model = models.load_model(filepath=filepath)
#     model.summary()
#     print('Loaded the model')


def load_model():
    if not os.path.exists(model_fn):
        os.system("wget " + model_url)
        print("saved: " + model_fn)


@app.route('/')
def index():
    return redirect('/static/index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.files and 'picfile' in request.files:
#         img = request.files['picfile'].read()
#         img = Image.open(io.BytesIO(img))
#         img.save('test.jpg')
#         img = np.asarray(img) / 255.
#         img = np.expand_dims(img, axis=0)
#         pred = model.predict(img)
#
#         players = [
#             'Lebron James',
#             'Stephen Curry',
#             'Kevin Durant',
#         ]
#
#         confidence = str(round(max(pred[0]), 3))
#         pred = players[np.argmax(pred)]
#
#         data = dict(pred=pred, confidence=confidence)
#         return jsonify(data)
#
#     return 'Picture info did not get saved.'

@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))

        img_fn = "test.png"
        img.save(img_fn)

        # os.system("python ../demo.py  %s %s" % (img_fn, model_fn))

        # os.system("cd ../")
        os.chdir("../")
        print(os.getcwd())
        # os.system("python ./demo.py  %s %s"
        #           % (os.path.join("docs", img_fn), os.path.join("docs", model_fn)))

        subprocess.call("python ./demo.py  %s %s"
                  % (os.path.join("docs", img_fn), os.path.join("docs", model_fn)), shell=True)

        os.chdir("./docs")

        # import time
        # time.sleep(10)

        # shutil.move("../demo_output", ".")

        data = dict(pred=0.5, confidence=0.5)
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.png', 'rb')
    data = fileob.read()
    return data


@app.route('/outsemseg', methods=['GET'])
def semseg_image():
    fileob = open('../demo_output/vis_test.png', 'rb')
    # fileob = open('./demo_output/vis_test.png', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    load_model()
    # model._make_predict_function()
    app.run(debug=False, port=5000)
