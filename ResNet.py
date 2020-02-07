import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os
import re
import gc
import random
import cv2
import matplotlib.image as mpimg
from math import gcd

strategy = tf.distribute.MirroredStrategy()

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                              kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                   momentum=bnMom)(inputs)

        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False,
                       padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                   momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("sigmoid")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model


ad_files = os.listdir("/home/k1651915/2D/AD/")
cn_files = os.listdir("/home/k1651915/2D/CN/")

sub_id_ad = []
sub_id_cn = [] 
for file in ad_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_ad:
        sub_id_ad.append(sub_id)

for file in cn_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_cn:
        sub_id_cn.append(sub_id)


def crop(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def get_images(folders, train = False):
    return_list = []
    for folder in folders:
        file_num_only = []
        os.chdir(folder)
        files = os.listdir('.')

        for png_file in files:
            file_num_only.append(re.search('(\\d*)', png_file).group(1))

        file_num_only.sort()
        png = mpimg.imread(file_num_only[87] + ".png")  # 87 corresponds to 88
        png = png[:, :, 1]
        png = crop(png)
        png = cv2.resize(png, (227, 227))
        if train:
            return_list.append(np.stack((png,) * 3, axis = 2))
            return_list = return_list + get_rotated_images(png)


        else:
            png = np.stack((png,) * 3, axis = 2)
            return_list.append(png)
        os.chdir('../')

    return return_list



def get_rotated_images(png):
    (h, w) = png.shape[:2]
    center = (w / 2, h / 2)
    
    angles = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
    rotated_pngs = []

    for angle in angles:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        r = cv2.warpAffine(png, m, (h, w))
        r = np.stack((r,) * 3, axis = 2)
        rotated_pngs.append(r)

    return rotated_pngs


results = []
seeds = range(1, 50)

for i in seeds:
    random.Random(i).shuffle(sub_id_ad)
    random.Random(i).shuffle(sub_id_cn)

    os.chdir("/home/k1651915/2D/AD/")
    ad_sub_train = sub_id_ad[0:111]
    ad_sub_validate = sub_id_ad[112:123]
    ad_sub_test = sub_id_ad[124:177]

    ad_sub_train_files = []
    ad_sub_validate_files = []
    ad_sub_test_files = []

    for file in ad_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in ad_sub_train:
            ad_sub_train_files.append(file)
        elif file_sub_id in ad_sub_validate:
            ad_sub_validate_files.append(file)
        elif file_sub_id in ad_sub_test:
            ad_sub_test_files.append(file)

    os.chdir("/home/k1651915/2D/AD")

    cn_sub_train = sub_id_cn[0:111]
    cn_sub_validate = sub_id_cn[112:123]
    cn_sub_test = sub_id_cn[124:177]

    cn_sub_train_files = []
    cn_sub_validate_files = []
    cn_sub_test_files = []

    for file in cn_files:
        file_sub_id = re.search('(OAS\\d*)', file).group(1)
        if file_sub_id in cn_sub_train:
            cn_sub_train_files.append(file)
        elif file_sub_id in cn_sub_validate:
            cn_sub_validate_files.append(file)
        elif file_sub_id in cn_sub_test:
            cn_sub_test_files.append(file)


    def equal_lists(list1, list2):
        if len(list2) % 2 != 0:
            list2.pop(len(list2) - 1)

        if len(list1) != len(list2):

            if len(list1) < len(list2):
                while len(list1) < len(list2):
                    list2.pop(len(list2) - 1)

            elif len(list1) > len(list2):
                while len(list1) > len(list2):
                    list1.pop(len(list1) - 1)

            return list1, list2


    x = equal_lists(cn_sub_train_files, ad_sub_train_files)

    os.chdir('/home/k1651915/2D/CN/')
    cn_train = get_images(x[0], True)
    os.chdir('/home/k1651915/2D/AD/')
    ad_train = get_images(x[1], True)
    train = np.asarray(cn_train + ad_train)

    y1 = np.zeros(len(cn_train))
    y2 = np.ones(len(ad_train))
    train_labels = np.concatenate((y1, y2), axis=None)

    print("train shape:")
    print(train.shape)
    print("train_labels shape:")
    print(train_labels.shape)

    x = None
    x = equal_lists(cn_sub_validate_files, ad_sub_validate_files)
    os.chdir('/home/k1651915/2D/CN/')
    cn_validate = get_images(x[0])
    os.chdir('/home/k1651915/2D/AD/')
    ad_validate = get_images(x[1])
    validate = np.asarray(cn_validate + ad_validate)

    y1 = np.zeros(len(x[0]))
    y2 = np.ones(len(x[1]))
    validation_labels = np.concatenate((y1, y2), axis=None)
    x = None

    cn_train = None
    ad_train = None
    cn_validate = None
    ad_validate = None
    gc.collect()

    #################################################

    with strategy.scope():
        model = ResNet.build(227, 227, 3, 1, (3, 4, 6, 8), (16, 32, 64, 128, 256, 512))
        model.compile(loss=tf.keras.losses.binary_crossentropy,
                        optimizer='adam',
                        metrics=['accuracy'])
        model.fit(train, train_labels,
                epochs = 30,
                batch_size = 20,
                validation_data = (validate, validation_labels))


    #################################################

    train = None
    validate = None
    gc.collect()

    x = equal_lists(cn_sub_test_files, ad_sub_test_files)
    os.chdir('/home/k1651915/2D/CN/')
    cn_test = get_images(x[0])
    os.chdir('/home/k1651915/2D/AD/')
    ad_test = get_images(x[1])

    test_data = cn_test + ad_test
    test_data = np.asarray(test_data)

    y1 = np.zeros(len(x[0]))
    y2 = np.ones(len(x[1]))
    test_labels = np.concatenate((y1, y2), axis=None)

    evaluation = model.evaluate(test_data, test_labels, verbose=0)

    cn_test = None
    ad_test = None
    test_data = None
    print(evaluation)
    results.append(evaluation[1])
    print("iteration and mean:")
    print(seed)
    print(sum(results)/len(results))
    gc.collect()
