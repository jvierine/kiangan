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

import sklearn
from sklearn.model_selection import train_test_split


tf.config.set_visible_devices([],'GPU')
def par2neprof(params, h=n.array([90,93,95,100,110,120,120,130,180,200,250,300,350,400,450,500])):
    """
    sum of two chapman function. convert chapman pars to electron density profile

    Model function for chapman electron 
    density profiles.
    
    Parameters  |   TYPE   | DESCRIPTION
    ------------------------------------------------
    h         | np array | Altitudes of electron density measurements
    
    E-region:
    ne_peak_E | float    | Maximum electron density value
    h_peak_E  | float    | Altitude at ne_peak
    w_Ed      | float    | Lower width of E-region
    w_Eu      | float    | Upper width of E-region
    
    F-Region:
    ne_peak_F | float    | Maximum electron density value
    h_peak_F  | float    | Altitude at ne_peak
    w_Fd      | float    | Lower width of F-region
    w_Fu      | float    | Upper width of F-region
    
    
    Return   |   TYPE   | DESCRIPTION
    ------------------------------------------------
    chap     | np array | Array with chapman profile of ne
    """
    ne_peak_E = params[0]
    h_peak_E  = params[1]
    w_Ed      = params[2]
    w_Eu      = params[3]
    
    ne_peak_F = params[4]
    h_peak_F  = params[5]
    w_Fd      = params[6]
    w_Fu      = params[7]
    
    # print(ne_peak_E)
    
    # Appending hights depending of ionospheric region widths
    hi_E = n.where(h <= h_peak_E, (h - h_peak_E)/(w_Ed +1), (h - h_peak_E)/w_Eu)
    hi_F = n.where(h <= h_peak_F, (h - h_peak_F)/(w_Fd +1), (h - h_peak_F)/w_Fu)
    
    # Two chapman electron density profile 
    chap = (ne_peak_E * n.exp(1 - hi_E - n.exp(-hi_E))) + (ne_peak_F*n.exp(1 - hi_F - n.exp(-hi_F)))
    
    return(chap,h)



if __name__ == "__main__":
    hd=h5py.File("training_data.h5","r")
    pars=hd["chapman_pars"][()]
    images=hd["images"][()]
    hd.close()

    training_images=tf.convert_to_tensor(images)

    hgts=n.array([90,93,95,100,110,120,120,130,180,200,250,300,350,400,450,500])
    n_hgts=len(hgts)
    ne_profiles=n.zeros([pars.shape[0],n_hgts])
    
    for pi in range(pars.shape[0]):
        ne,h=par2neprof(pars[pi,:])
        ne[ne<=0]=1e8
        ne_profiles[pi,:]=n.log10(ne)

    # avoid too small values
    ne_profiles[ne_profiles<8]=8

    #print(training_images.shape)
    #print(ne_profiles.shape)
    #train_images, val_images, train_data, val_data = train_test_split(training_images, ne_profiles, test_size=0.20, shuffle=True)

    val_images=training_images[0:100,:,:]
    train_images=training_images[100:,:,:]    
    val_data=ne_profiles[0:100,:]
    train_data=ne_profiles[100:,:]
#    print(train_data.shape)
 #   print(val_data.shape)
  #  print(train_images.shape)
   # print(val_images.shape)        
    
    training_images=tf.convert_to_tensor(train_images)
    training_data=tf.convert_to_tensor(train_data)    
    val_images=tf.convert_to_tensor(val_images)
    val_data=tf.convert_to_tensor(val_data)    


    input_shape=(81, 81, 1)
    
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output for dense layers
    model.add(layers.Flatten())
    
    # Dense layers (encoder)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(16, activation='linear'))

    model.compile(optimizer="adam", loss="mse")  # Use appropriate loss function for regression
    
    monitor="val_loss"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="ionosonde_gan",
                                                          monitor=monitor,
                                                          save_best_only=True)
    model_fit =  model.fit(training_images,
                           training_data,
                           epochs=100,
                           batch_size=32,
                           validation_data=(val_images, val_data),
                           callbacks=[model_checkpoint])

    Cost =  model_fit.history["loss"]

    plt.plot(Cost)
    plt.show()
    
    

        
