import numpy as np 
import tensorflow as tf 

def simple_unet(input_size, n_filt):
    #Build the model
    inputs = tf.keras.layers.Input(input_size)
    
    #Contraction path
    conv1 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = tf.keras.layers.Dropout(0.1)(conv1)
    conv1 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.1)(conv2)
    conv2 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
     
    conv3 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.1)(conv3)
    conv3 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(16*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = tf.keras.layers.Dropout(0.3)(conv5)
    conv5 = tf.keras.layers.Conv2D(16*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    
    #Expansive path 
    up6   = tf.keras.layers.Conv2DTranspose(8*n_filt, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6   = tf.keras.layers.concatenate([up6, conv4])
    conv6 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
     
    up7   = tf.keras.layers.Conv2DTranspose(4*n_filt, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7   = tf.keras.layers.concatenate([up7, conv3])
    conv7 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = tf.keras.layers.Dropout(0.1)(conv7)
    conv7 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    
    up8 = tf.keras.layers.Conv2DTranspose(2*n_filt, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2])
    conv8 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = tf.keras.layers.Dropout(0.1)(conv8)
    conv8 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
     
    up9 = tf.keras.layers.Conv2DTranspose(n_filt, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1])
    conv9 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = tf.keras.layers.Dropout(0.1)(conv9)
    conv9 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])

    return model