from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import Regularizer
import matplotlib.pyplot as plt
import numpy

def NIN(activation,  HE, BN, filename):
    print(activation, HE, BN)
    batch_size = 128
    num_classes = 10
    epochs = 164
    data_augmentation = True
    ## load weight or not
    load = False 
    
    # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR10 images are RGB.
    img_channels = 3
    
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    
    print(x_train.shape[1:])
    
    bias_weight = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    
    if HE == True:
        conv_weight = keras.initializers.he_normal(seed=None)
    else:
        conv_weight = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        
    if activation == 'leaky':
        print('leaky')
        leaky =keras.layers.advanced_activations.LeakyReLU(alpha=0.01)
    
    model = Sequential()
    
    model.add(Conv2D(192,(5, 5) ,padding='same', input_shape=x_train.shape[1:],kernel_initializer = conv_weight,bias_initializer = bias_weight,kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    model.add(Conv2D(160, (1, 1), padding='same',kernel_initializer = conv_weight, bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(Conv2D(96,(1, 1) ,padding='same',kernel_initializer = conv_weight,bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same'))
    
    model.add(Dropout(0.5))
    print(model.layers[-1].output_shape)
    
    model.add(Conv2D(192, (5, 5), padding='same',kernel_initializer = conv_weight,bias_initializer=bias_weight,kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(Conv2D(192, (1, 1),padding='same',kernel_initializer = conv_weight,bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(Conv2D(192, (1, 1),padding='same',kernel_initializer = conv_weight,bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding='same'))
    
    model.add(Dropout(0.5))
    print(model.layers[-1].output_shape)
    
    
    model.add(Conv2D(192, (3, 3), padding='same',kernel_initializer = conv_weight,bias_initializer=bias_weight,kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
        
    model.add(Conv2D(192, (1, 1),padding='same',kernel_initializer = conv_weight,bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(Conv2D(10, (1, 1),padding='same',kernel_initializer = conv_weight,bias_initializer='zeros',kernel_regularizer=keras.regularizers.l2(0.0001)))
    if BN == True :
        model.add(keras.layers.normalization.BatchNormalization())
    if activation == 'leaky':
        model.add(leaky)
    else:
        model.add(Activation(activation))
    
    model.add(AveragePooling2D(pool_size=(8, 8)))
    
    model.add(Flatten())
    
    model.add(Activation('softmax'))
    print(model.layers[-1].output_shape)
    
    ##load 
    if load == True :
        model.load_weights('lab1.h5')
    ##learning Rate Scheduler
    def step_decay(epoch):
        initial_lrate = 0.1
        lrate = initial_lrate
        if (epoch >= 40):
            lrate = 0.05
        if (epoch >= 80):
            lrate = 0.01
        if (epoch >= 100):
            lrate = 0.005
        if (epoch >= 120):
            lrate = 0.001
    
        print(lrate,epoch)
        return lrate
        
        
    sgd = SGD(lr=0.0, decay=0, momentum=0.9, nesterov=True)
    # Let's train the model using SGD with momentum
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #print(x_train[:,:,:,1])
    
    x_train[:,:,:,0]=(x_train[:,:,:,0]-125.3)/63.0
    x_train[:,:,:,1]=(x_train[:,:,:,1]-123.0)/62.1
    x_train[:,:,:,2]=(x_train[:,:,:,2]-113.9)/66.7
    
    x_test[:,:,:,0]=(x_test[:,:,:,0]-125.3)/63.0
    x_test[:,:,:,1]=(x_test[:,:,:,1]-123.0)/62.1
    x_test[:,:,:,2]=(x_test[:,:,:,2]-113.9)/66.7
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]
        
        history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                            samples_per_epoch = x_train.shape[0], 
    			                  nb_epoch = epochs,
                            validation_data = (x_test, y_test),
                            callbacks=callbacks_list)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test accuracy:', score[1])
    #model.save_weights(filename+'.h5')
    
    f = open('log.txt', 'a')
    
    f.write(filename)
    f.write(' : ')
    f.write(str(score[1]))
    f.write('\n')
    print(history.history.keys())
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename+'_acc.png')
    plt.clf()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename+'_loss.png')
    plt.clf()
