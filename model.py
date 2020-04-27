import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

class Model:
    def __init__(self):
        self.base_dir = 'data/'
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')
        self.train_humans_dir = os.path.join(self.train_dir, 'humans')
        self.train_other_dir = os.path.join(self.train_dir, 'other')
        self.validation_humans_dir = os.path.join(self.validation_dir, 'human')
        self.validation_other_dir = os.path.join(self.validation_dir, 'other')

        self.conv_base = Xception(weights='imagenet', 
                             include_top=False, 
                             input_shape=(300, 300, 3))

        self.conv_base.trainable = False
        self.model = keras.models.Sequential()
        self.model.add(self.conv_base)
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def load_data(self):
        train_datagen = ImageDataGenerator(
                        rescale=1./255, 
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'
                        )
        
        self.train_generator = train_datagen.flow_from_directory(
                            self.train_dir,
                            target_size=(150, 150),
                            batch_size=20,
                            class_mode='binary')
        
        validation_datagen = ImageDataGenerator(rescale=1./255)

        self.validation_generator = validation_datagen.flow_from_directory(
                                self.validation_dir,
                                target_size=(150, 150),
                                batch_size=20,
                                class_mode='binary')
        
    def build_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

        # train
        history = self.model.fit_generator(self.train_generator, steps_per_epoch=100, epochs=100,
                                      validation_data=self.validation_generator, validation_steps=50)


def main():
    model = Model()
    model.load_data()
    model.build_model()



if __name__=='__main__':
    main()