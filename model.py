import os
import glob
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator # causing errors
from tensorflow.keras import optimizers


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

base_dir = 'data\\'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_humans_dir = os.path.join(train_dir, 'humans')
train_other_dir = os.path.join(train_dir, 'other')
validation_humans_dir = os.path.join(validation_dir, 'human')
validation_other_dir = os.path.join(validation_dir, 'other')

# model paths
model_dir = os.path.dirname('models\\')
checkpoint_path = os.path.join(model_dir, 'checkpoints/')
log_dir = os.path.join(base_dir, 'logs\\')

def create_model():
    conv_base = Xception(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    conv_base.trainable = False
    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

    return model

def train(model):
    # data loaders
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
        
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300),
                                                        batch_size=20, class_mode='binary')
        
    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(300, 300),
                                                                  batch_size=20, class_mode='binary')
    
    # checkpoints
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # tensorboard
    log = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log, histogram_freq=10)

    # callbacks
    callbacks = [checkpoint_callback, tensorboard_callback]

    # train
    history = model.fit_generator(train_generator, steps_per_epoch=25, epochs=10,
                                  validation_data=validation_generator, validation_steps=25,
                                  callbacks=callbacks)

    return model


def data_loader():
    #classes = np.array([item.name for item in train_dir.glob('*')])
    print(glob.glob(train_dir + str('*')))
    #print(classes)

def main():
    model = create_model()
    train(model)

if __name__=='__main__':
    main()