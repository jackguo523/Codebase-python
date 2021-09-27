# adopted from Sreenivas Bhattiprolu
# modified by Jiaming Guo


import tensorflow as tf
import numpy as np
import glob
import h5py

from skimage.io import imread,imsave
from skimage.transform import resize

IMG_WIDTH=128
IMG_HEIGHT=128
IMG_CHANNELS=3

x_train_path='jack_data/x_train'
y_train_path='jack_data/y_train'
x_test_path='jack_data/x_test'
y_test_path='jack_data/y_test'

x_train_names=glob.glob(x_train_path+'/*.tif')
y_train_names=glob.glob(y_train_path+'/*.tif')
x_test_names=glob.glob(x_test_path+'/*.tif')
y_test_names=glob.glob(y_test_path+'/*.tif')

x_train=np.zeros((len(x_train_names),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
y_train=np.zeros((len(x_train_names),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
x_test=np.zeros((len(x_test_names),IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
y_test=np.zeros((len(x_test_names),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
for i in range(len(x_train_names)):
    tmp1=imread(x_train_names[i])
    tmp1=resize(tmp1,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    x_train[i]=tmp1
    
    tmp2=imread(y_train_names[i])[:,:,1]
    tmp2=np.expand_dims(resize(tmp2,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True),axis=-1)
    tmp2=np.maximum(tmp2,np.zeros((IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool))
    y_train[i]=tmp2
    
for i in range(len(x_test_names)):
    tmp3=imread(x_test_names[i])
    tmp3=resize(tmp3,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    x_test[i]=tmp3
    
    tmp4=imread(y_test_names[i])[:,:,1]
    tmp4=np.expand_dims(resize(tmp4,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True),axis=-1)
    tmp4=np.maximum(tmp4,np.zeros((IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool))
    y_test[i]=tmp4


# start the model
inputs=tf.keras.layers.Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
s=tf.keras.layers.Lambda(lambda x:x/255)(inputs)

# contracting path
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(s)
c1=tf.keras.layers.Dropout(0.1)(c1)
c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1)
c2=tf.keras.layers.Dropout(0.1)(c2)
c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
p2=tf.keras.layers.MaxPooling2D((2,2))(c2)
 
c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
c3=tf.keras.layers.Dropout(0.2)(c3)
c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
p3=tf.keras.layers.MaxPooling2D((2,2))(c3)
 
c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
c4=tf.keras.layers.Dropout(0.2)(c4)
c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
p4=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c4)
 
c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
c5=tf.keras.layers.Dropout(0.3)(c5)
c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)

# expanding path 
u6=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2, 2),padding='same')(c5)
u6=tf.keras.layers.concatenate([u6,c4])
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6)
c6=tf.keras.layers.Dropout(0.2)(c6)
c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
 
u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7=tf.keras.layers.concatenate([u7,c3])
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
c7=tf.keras.layers.Dropout(0.2)(c7)
c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)
 
u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8=tf.keras.layers.concatenate([u8,c2])
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
c8=tf.keras.layers.Dropout(0.1)(c8)
c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
 
u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2, 2),padding='same')(c8)
u9=tf.keras.layers.concatenate([u9,c1],axis=3)
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
c9=tf.keras.layers.Dropout(0.1)(c9)
c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)
 
outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)
 
model=tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

filepath="vessel.h5" # initial new weights

earlystopper=tf.keras.callbacks.EarlyStopping(patience=3,verbose=1)

checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

callbacks_list=[earlystopper,checkpoint]

history=model.fit(x_train,y_train,validation_split=0.1,batch_size=16,epochs=25,callbacks=callbacks_list)

#h5f=h5py.File('vessel.h5','w')
model.load_weights('vessel.h5') # load best weights
model.save_weights('rep_vessel.h5') # save best weights
_y_test=model.predict(x_test)
_y_test=(_y_test>=0.5).astype(np.uint8)
for i in range(len(_y_test)):
    imsave('jack_data/1/{:03d}.tif'.format(i+1),_y_test[i,:,:,0]*255)