import os
import glob
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator # causing errors
from tensorflow.keras import optimizers


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# data paths
base_dir = 'data\\'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_humans_dir = os.path.join(train_dir, 'humans')
train_other_dir = os.path.join(train_dir, 'other')
validation_humans_dir = os.path.join(validation_dir, 'human')
validation_other_dir = os.path.join(validation_dir, 'other')

# model paths
model_dir = os.path.dirname('models\\')
checkpoint_path = os.path.join(model_dir, 'checkpoints\\')
log_dir = os.path.dirname('logs\\')


class Model(tf.keras.models.Sequential):
    def __init__(self, output_layers=[], learning_rate=2e-5, epochs=100):
        super(Model, self).__init__()
        self.file_name = 'Xception' + str(epochs) + '.epochs.' + str(learning_rate) + '.lr.' 
        self.epochs = epochs
        # add convolutional base
        conv_base = Xception(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        conv_base.trainable = True

        self.add(conv_base)
        self.add(keras.layers.Flatten())
        
        # output layers
        for layer_dim in output_layers:
            self.add(keras.layers.Dense(layer_dim, activation='relu'))
            self.file_name = self.file_name + '_' + str(layer_dim)

        self.add(keras.layers.Dense(1, activation='sigmoid'))
        
        self.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['acc'])

    def train(self):
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
        ckpt_path = checkpoint_path + 'backup.ckpt'
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_weights_only=True,
                                                                verbose=1)

        # tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        # callbacks
        callbacks = [checkpoint_callback, tensorboard_callback]

        # train
        history = self.fit_generator(train_generator, steps_per_epoch=25, epochs=self.epochs,
                                    validation_data=validation_generator, validation_steps=25,
                                    callbacks=callbacks)


        model.save(os.path.join(model_dir, self.file_name))

    def load_model(checkpoint_path):
        model = create_model()
        model.load_weights(checkpoint_path)
        return model

def main():
    model = Model(output_layers=[32], epochs=100)
    model.train()

    model = Model(output_layers=[64], epochs=100)
    model.train()

    model = Model(output_layers=[128], epochs=100)
    model.train()

    model = Model(output_layers=[256], epochs=100)
    model.train()

    model = Model(output_layers=[512], epochs=100)
    model.train()

    model = Model(output_layers=[1024], epochs=100)
    model.train()

    # two output layers
    model = Model(output_layers=[32, 8], epochs=100)
    model.train()

    model = Model(output_layers=[64, 8], epochs=100)
    model.train()

    model = Model(output_layers=[128, 8], epochs=100)
    model.train()

    model = Model(output_layers=[256, 8], epochs=100)
    model.train()

    model = Model(output_layers=[512, 8], epochs=100)
    model.train()

    model = Model(output_layers=[1024, 8], epochs=100)
    model.train()

    print('done')

if __name__=='__main__':
    main()