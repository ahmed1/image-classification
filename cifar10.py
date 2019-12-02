from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers import Add, Flatten, AveragePooling2D, Dense
from keras.models import Model

# This is one res_layer which we can use to create a deep neural network
def res_layer(input_layer, n=64):
    L1 = Conv2D(n, (3, 3), padding='same', activation='relu')(input_layer)
    L2 = Conv2D(n, (3, 3), padding='same', activation='relu')(L1)
    L2 = Conv2D(n, (3, 3), padding='same', activation='relu')(L2)
    L3 = Add()([L2, input_layer])
    return L3


# size of cifar10 dataset images
main_input = Input(shape=(32,32,3))
#all the layers of the res-net
L1 = Conv2D(64, (7, 7), strides=(2,2), padding='same', activation='relu')(main_input)
L2 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(L1)

L2 = res_layer(L2)
L2 = res_layer(L2)
L2 = res_layer(L2)

L3 = AveragePooling2D()(L2)

L3 = Flatten()(L3)
L4 = Dense(256)(L3)
L5 = Dense(10,activation='softmax')(L4)

#Model function from Keras shows map neural network including output shape of each layer and the number of parameters
model = Model(main_input, L5)
model.summary()


#load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#normalize the input data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#Optimizer functino used with hyperparameters such as learning rate, decay, and momentum
sgd = optimizers.SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=15, shuffle='batch', batch_size=64)
f = open('log.txt','w')


test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

# There is clearly both a high bias and high variance here -- will work to change this in the future
