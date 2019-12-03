from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Add, Flatten, AveragePooling2D, Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras import optimizers
from keras.models import Model, Sequential
from keras.regularizers import l2
import time
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import print_summary, to_categorical
from keras import backend as K
import sys
import os
import numpy as np
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

def res_layer(input_layer, n=16, strides=1):
    L1 = Conv2D(n, (3, 3), padding='same', strides=strides)(input_layer)
    L1 = BatchNormalization()(L1)
    L1 = Activation('relu')(L1)
    L2 = Conv2D(n, (3, 3), padding='same')(L1)
    L2 = BatchNormalization()(L2)
    L2 = Activation('relu')(L2)
    L3 = Conv2D(n, (3, 3), padding='same')(L2)
    L3 = BatchNormalization()(L3)
    L4 = Add()([L3, L1])
    L4 = Activation('relu')(L4)
    return L4
    
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


batch_size = 32
nb_epoch = 250
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar100.csv')


main_input = Input(shape=(32,32,3))
L2 = res_layer(main_input, 16)
L2 = res_layer(L2, 16)
L2 = res_layer(L2, 32, strides=2)
L2 = res_layer(L2, 32)
L2 = res_layer(L2, 64, strides=2)
L2 = res_layer(L2, 64)
L3 = AveragePooling2D()(L2)
L3 = Flatten()(L3)
L4 = Dense(100,activation='softmax')(L3)



model = Model(main_input, L4)
model.summary()


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



sgd = optimizers.SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizers.Adam(lr=lr_schedule(0)),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
datagen.fit(x_train)


start_time = time.time()
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=nb_epoch,
                    verbose=1,
                    callbacks=[LearningRateScheduler(lr_schedule), csv_logger])
                    # callbacks=[LearningRateScheduler(lr_schedule), early_stopper, csv_logger])
end_time = time.time()



test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)


print(end_time-start_time)
