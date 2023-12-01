import numpy as n
import matplotlib.pyplot as plt
import glob
import h5py

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras import layers, models, optimizers
from tensorflow import keras
import tensorflow as tf

import ionosonde_gan as ig

tf.config.set_visible_devices([],'GPU')

model=tf.keras.models.load_model("ionosonde_gan")

hd=h5py.File("/media/j/fee7388b-a51d-4e10-86e3-5cabb0e1bc13/Kian/training_data.h5","r")
pars=hd["chapman_pars"][()]
images=hd["images"][()]
hd.close()


for i in range(images.shape[0]):

    hgts=n.array([90,93,95,100,110,120,120,130,180,200,250,300,350,400,450,500])

    print(images.shape)
    im_in=images[i,:,:]
    im_in.shape=(1,81,81,1)
    ne_prof=model.predict(im_in)[0]

    
    plt.semilogx(10**ne_prof,hgts,label="cnn")

    real_ne,real_h=ig.par2neprof(pars[i,:])
    plt.semilogx(real_ne,real_h,label="real")
    plt.legend()

    plt.show()
