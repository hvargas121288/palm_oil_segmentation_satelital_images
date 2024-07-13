import numpy as np 
import tensorflow as tf 

def simple_unet(input_size, n_filt):
    #Build the model
    inputs = tf.keras.layers.Input(input_size)
    
    #Contraction path
    conv1 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 = tf.keras.layers.Dropout(0.25)(conv1)
    conv1 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = tf.keras.layers.Dropout(0.25)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.25)(conv2)
    conv2 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    conv2 = tf.keras.layers.Dropout(0.25)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
     
    conv3 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.25)(conv3)
    conv3 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    conv3 = tf.keras.layers.Dropout(0.25)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.25)(conv4)
    conv4 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    conv4 = tf.keras.layers.Dropout(0.25)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)


    conv5 = tf.keras.layers.Conv2D(16*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = tf.keras.layers.Dropout(0.25)(conv5)
    conv5 = tf.keras.layers.Conv2D(16*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    conv5 = tf.keras.layers.Dropout(0.25)(conv5)
    
    #Expansive path 
    up6   = tf.keras.layers.Conv2DTranspose(8*n_filt, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6   = tf.keras.layers.concatenate([up6, conv4])
    conv6 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up6)
    conv6 = tf.keras.layers.Dropout(0.25)(conv6)
    conv6 = tf.keras.layers.Conv2D(8*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = tf.keras.layers.Dropout(0.25)(conv6)
     
    up7   = tf.keras.layers.Conv2DTranspose(4*n_filt, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7   = tf.keras.layers.concatenate([up7, conv3])
    conv7 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up7)
    conv7 = tf.keras.layers.Dropout(0.25)(conv7)
    conv7 = tf.keras.layers.Conv2D(4*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = tf.keras.layers.Dropout(0.25)(conv7)
    
    up8 = tf.keras.layers.Conv2DTranspose(2*n_filt, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2])
    conv8 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up8)
    conv8 = tf.keras.layers.Dropout(0.25)(conv8)
    conv8 = tf.keras.layers.Conv2D(2*n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
    conv8 = tf.keras.layers.Dropout(0.25)(conv8)
     
    up9 = tf.keras.layers.Conv2DTranspose(n_filt, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up9)
    conv9 = tf.keras.layers.Dropout(0.25)(conv9)
    conv9 = tf.keras.layers.Conv2D(n_filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
    conv9 = tf.keras.layers.Dropout(0.25)(conv9)
    
    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(conv9)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[out_mask,out_edge])

    return model