"""
deactivate
conda.bat deactivate
LungUNETCPUEnv\Scripts\activate
python lungunetmodel.py
"""
import numpy as np # linear algebra
import os
import cv2
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

model = unet(input_size=(512,512,1))
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, 'binary_accuracy'])
# model.summary()

weight_path="cxr_reg_weights.best.hdf5"
model_weights_path = os.path.join(ROOT_DIR,"Weights",weight_path)
model.load_weights(model_weights_path)

"""
Shapes that you wish to resize to
"""

Shape_X,Shape_Y=512,512

def read_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image,(Shape_Y,Shape_X))
    return image


def get_preds(image):
    prep_unet_input_img_1 = image.reshape(1,Shape_X,Shape_Y,1)
    prep_unet_input_img = (prep_unet_input_img_1-127.0)/127.0
    pred_img = model.predict(prep_unet_input_img)
    pred_img_preprocessed_1 = np.squeeze(pred_img)
    pred_img_preprocessed = (pred_img_preprocessed_1*255>127).astype(np.int8)
    res = cv2.bitwise_and(image,image,mask = pred_img_preprocessed)
    return res,pred_img_preprocessed
cv2.imwrite(save_path,res)


if __name__ == '__main__':
    INP = os.path.join(ROOT_DIR,"Sample_Inputs")
    INP_RESHAPED = os.path.join(ROOT_DIR,"Sample_Inputs_Reshaped")
    RES = os.path.join(ROOT_DIR,"Sample_Masked_Results")
    MASK_PATH = os.path.join(ROOT_DIR,"Sample_Lung_Masks")
    
    
    create_folders([INP,INP_RESHAPED,RES])
    """
    Images Output :
    Original Reshaped Image
    Superimposed Lungs Segmentation
    """
    input_files = os.listdir(INP)
    for i,f in enumerate(input_files):
        img = read_image(os.path.join(INP,f))
        reshaped_img = deepcopy(img)
        segmented_output,mask = get_preds(reshaped_img)
        
        cv2.imwrite(os.path.join(INP_RESHAPED,f),reshaped_img )
        cv2.imwrite(os.path.join(RES,f),segmented_output)
        cv2.imwrite(os.path.join(MASK_PATH ,f),mask)