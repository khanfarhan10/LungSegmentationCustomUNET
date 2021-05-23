"""
deactivate
conda.bat deactivate
CarUNETCPUEnv\Scripts\activate
python app.py
"""
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, send_file
import os
import zipfile


import colorsys
import random
from copy import deepcopy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# %env SM_FRAMEWORK=tf.keras
os.environ['SM_FRAMEWORK'] = "tf.keras"

import cv2
import tensorflow as tf
# import keras
import numpy as np

MAIN_SIZE_X = 512
MAIN_SIZE_Y = 512

       

import segmentation_models as sm

BACKBONE = 'resnet34' #'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['scratch']
LR = 0.0001

# "templates" this is for plain html files or "Great_Templates" this is for complex css + imgs +js +html+sass
TEMPLATES = "templates"

app = Flask(__name__, static_folder="assets", template_folder=TEMPLATES)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB Standard File Size
# ROOT_DIR = os.getcwd()
# ROOT_DIR = app.instance_path
ROOT_DIR = app.root_path
# Reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1
activation = 'sigmoid'


weight_path="Keras_SegModels_UNET_Car_Damage_detectioneps300.hdf5"
weight_path = os.path.join(ROOT_DIR,"Weights",weight_path)

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation,weights=weight_path,encoder_weights =None)
"""
encoder_weights – one of None (random initialization), imagenet (pre-training on ImageNet).
"""

# define optomizer
optim = tf.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

model.load_weights(weight_path)

def read_image(img_path):
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(MAIN_SIZE_Y,MAIN_SIZE_X))
    return image

def get_preds(image):
    pr_mask = model.predict(image).round()
    return pr_mask[..., 0].squeeze()

def props(arr,u=0):
    print("Shape :",arr.shape,"Maximum :",arr.max(),"Minimum :",arr.min(),"Data Type :",arr.dtype,end=' ')
    if u==1:
        print("Unique Values :",np.unique(arr),end=' ')
    print()

def create_folders(lst):
    for folder in lst:
        os.makedirs(folder, exist_ok=True)
        

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def apply_model(img=None,img_path=None):
    """
    reshaped_img,color_res,maskclass,maskbg,segmentedclassres,segmentedbgres = apply_model(img=None,img_path=None)
    """
    if img is None:
        img = read_image(img_path)
    reshaped_img = deepcopy(img)
    """
    Shape_X,Shape_Y,Shape_Z = img.shape
    img = img.reshape(1,Shape_X,Shape_Y,Shape_Z)
    """
    exp_img = np.expand_dims(img, axis=0)
    inv_mask = get_preds(exp_img)
    inv_mask = inv_mask.astype(bool)
    mask = np.invert(inv_mask)
    mask = mask.astype(np.uint8)
    # props(mask,u=1)
    segmentedclassres = cv2.bitwise_and(img,img,mask = mask)
    segmentedbgres = cv2.bitwise_and(img,img,mask = inv_mask.astype(np.uint8))
    
    """
    partial superimposing
    """
    colours = random_colors(N=1, bright=True)
    res= apply_mask(img, mask, color=colours[0], alpha=0.5)
    return reshaped_img ,res,mask*255,inv_mask.astype(np.uint8)*255,segmentedclassres,segmentedbgres




@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    


FileSaveDir = os.path.join(ROOT_DIR, "TempSaved")
ImgDir = os.path.join(ROOT_DIR, "Experimental_Imgs")
import shutil

@app.route('/uploadsuccess', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # f.save(f.filename)
        # Perform Some File Validation so that only DOCX File Can be uploaded
        if os.path.exists(ImgDir):
            shutil.rmtree(ImgDir)
        FileSavePath = os.path.join(FileSaveDir, f.filename)
        os.makedirs(FileSaveDir, exist_ok=True)
        os.makedirs(ImgDir, exist_ok=True)
        ImgSavePath = os.path.join(ImgDir, f.filename)
        f.save(ImgSavePath)

        reshaped_img, color_res, maskclass, maskbg, segmentedclassres, segmentedbgres = apply_model(img_path=ImgSavePath)
        cv2.imwrite(os.path.join(FileSaveDir, "reshaped_img.png"), reshaped_img)
        cv2.imwrite(os.path.join(FileSaveDir, "color_res.png"), color_res)
        cv2.imwrite(os.path.join(FileSaveDir, "maskclass.png"), maskclass)
        cv2.imwrite(os.path.join(FileSaveDir, "maskbg.png"), maskbg)
        cv2.imwrite(os.path.join(FileSaveDir, "segmentedclassres.png"), segmentedclassres)
        cv2.imwrite(os.path.join(FileSaveDir, "segmentedbgres.png"), segmentedbgres)
        
        NewFileSaveDir = os.path.join(ROOT_DIR, "assets","images")
        cv2.imwrite(os.path.join(NewFileSaveDir, "reshaped_img.png"), reshaped_img)
        cv2.imwrite(os.path.join(NewFileSaveDir, "color_res.png"), color_res)
        return render_template('results.html')
    return render_template('results.html')

    


@app.route('/results')
def upload_excel_file():
    return render_template('results.html')


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def zipper(dir_path, zip_path):
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    zipdir(dir_path, zipf)
    zipf.close()


@app.route('/download')
def return_files_tut():
    # ZipPath = os.path.join(FileSaveDir, "CarDamageDetectionResults.zip")
    ZipPath = "CarDamageDetectionResults.zip"
    zipper(FileSaveDir, ZipPath)

    return send_file(ZipPath, as_attachment=True, mimetype='application/zip',
                     attachment_filename='CarDamageDetectionResults.zip')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'assets', 'favicons'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    app.run() # debug=True
