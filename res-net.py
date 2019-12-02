from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers import Add, Flatten, AveragePooling2D, Dense
from keras.models import Model
from keras import optimizers
import time


def res_layer(input_layer, n=64):
L1 = Conv2D(n, (3, 3), padding='same', activation='relu')(input_layer)
L2 = Conv2D(n, (3, 3), padding='same', activation='relu')(L1)
L2 = Conv2D(n, (3, 3), padding='same', activation='relu')(L2)
L3 = Add()([L2, input_layer])
return L3


main_input = Input(shape=(32,32,3))
L1 = Conv2D(64, (7, 7), strides=(2,2), padding='same', activation='relu')(main_input)
L2 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(L1)

L2 = res_layer(L2)
L2 = res_layer(L2)
L2 = res_layer(L2)

L3 = AveragePooling2D()(L2)

L3 = Flatten()(L3)
L4 = Dense(256)(L3)
L5 = Dense(10,activation='softmax')(L4)


model = Model(main_input, L5)
print(model.summary())



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
sgd = optimizers.SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
start_time = time.time()
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
end_time = time.time()
history = model.fit(x_train, y_train, epochs=3, shuffle='batch', batch_size=64)
f = open('log.txt','w')


test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)
