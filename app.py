"""
deactivate
conda.bat deactivate
LungUNETCPUEnv\Scripts\activate
python app.py
"""
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, send_file
import os
import zipfile

import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras

import random
from copy import deepcopy

import os
import cv2
import tensorflow as tf
# import keras
import numpy as np


       
def props(arr,u=0):
    print("Shape :",arr.shape,"Maximum :",arr.max(),"Minimum :",arr.min(),"Data Type :",arr.dtype,end=' ')
    if u==1:
        print("Unique Values :",np.unique(arr),end=' ')
    print()

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.flatten(y_true)
    y_pred_f = tf.keras.flatten(y_pred)
    intersection = tf.keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.sum(y_true_f) + tf.keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = tf.keras.layers.Input(input_size)
    
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # up6 = tf.keras.layers.Concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = tf.concat([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    # up7 = tf.keras.layers.Concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = tf.concat([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    # up8 = tf.keras.layers.Concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = tf.concat([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    # up9 = tf.keras.layers.Concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = tf.concat([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return tf.keras.Model(inputs=[inputs], outputs=[conv10])

model = unet(input_size=(512,512,1))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'binary_accuracy'])
# model.summary()
ROOT_DIR = os.getcwd()
weight_path="cxr_reg_weights.best.hdf5"
model_weights_path = os.path.join(ROOT_DIR,"Weights",weight_path)
model.load_weights(model_weights_path)

"""
Shapes that you wish to resize to
"""

Shape_X,Shape_Y=512,512

def read_image(img_path):
    image = cv2.imread(img_path,0)
    image = cv2.resize(image,(Shape_Y,Shape_X))
    return image


def get_preds(image):
    prep_unet_input_img_1 = image.reshape(1,Shape_X,Shape_Y,1)
    prep_unet_input_img = (prep_unet_input_img_1-127.0)/127.0
    pred_img = model.predict(prep_unet_input_img)
    pred_img_preprocessed_1 = np.squeeze(pred_img)
    pred_img_preprocessed = (pred_img_preprocessed_1*255>127).astype(np.int8)
    res = cv2.bitwise_and(image,image,mask = pred_img_preprocessed)
    return res,pred_img_preprocessed*255

def create_folders(lst):
    for folder in lst:
        os.makedirs(folder, exist_ok=True)

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
        reshaped_img = read_image(ImgSavePath)
        
        segmented_output,mask = get_preds(reshaped_img)
        
        cv2.imwrite(os.path.join(FileSaveDir, "reshaped_img.png"), reshaped_img)
        cv2.imwrite(os.path.join(FileSaveDir, "binarymask.png"), mask)
        cv2.imwrite(os.path.join(FileSaveDir, "segmentedlungmask.png"), segmented_output)
 
        
        NewFileSaveDir = os.path.join(ROOT_DIR, "assets","images")
        cv2.imwrite(os.path.join(NewFileSaveDir, "reshaped_img.png"), reshaped_img)
        cv2.imwrite(os.path.join(NewFileSaveDir, "color_res.png"), segmented_output)
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
    ZipPath = "LungMasksDetectionResults.zip"
    zipper(FileSaveDir, ZipPath)

    return send_file(ZipPath, as_attachment=True, mimetype='application/zip',
                     attachment_filename=ZipPath)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'assets', 'favicons'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    app.run() # debug=True
