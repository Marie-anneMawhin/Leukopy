import streamlit as st

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd

from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image


## PARAMETERS

batch_size = 32
img_height = 360
img_width  = 360
classes = ["BA","BNE","EO","ERB","LY","MMY","MO","MY","PLT","PMY","SNE"]
label_map = {'BA': 0, 'BNE': 1, 'EO': 2, 'ERB': 3, 'LY': 4, 'MMY': 5, 'MO': 6, 'MY': 7, 'PLT': 8, 'PMY': 9, 'SNE': 10}

vgg19_path = Path("./data/model/vgg19/weights")


#GRAD-CAM :

def make_heatmap(img_array, model, last_conv_layer, class_index):

    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))

    heatmap_tmp = last_conv_layer_output[0].numpy()

    # Multiplie chaque carte d'activation par le gradient, puis moyenne
    for i in range(last_conv_layer_output.shape[3]):
        heatmap_tmp[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(heatmap_tmp, axis=-1)

    return heatmap


def gradcam(model, img, img_orig, 
            img_height, img_width, class_index, 
            alpha = 0.5):
    
    ## Calcul de la Heatmap :
    # Désactive softmax sur la dernière couche :
    model.layers[-1].activation = None    
    # Détecte la dernière couche de convolution du modèle :
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = model.get_layer(layer.name)
            break
    # Calcul
    heatmap = make_heatmap(img, model, last_conv_layer, class_index)
    # Réactive softmax :
    model.layers[-1].activation = tf.keras.activations.softmax


    ## Traitement de la Heatmap :
    # Applique ReLu (élimine les valeurs négatives de la heatmap)
    heatmap = np.maximum(0, heatmap)
    # Normalisation
    heatmap = heatmap/heatmap.max()
    
    
    ## Superposition de l'image "img_orig" et de la heatmap
    # 1/ Rescale heatmap: 0-255
    heatmap = np.uint8(255*heatmap)
    # 2/ Jet colormap
    jet = cm.get_cmap("jet")
    # 3/ Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # 4/ Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_orig.shape[1], img_orig.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    # 6/ Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha + img_orig
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

# Load model :
#@st.cache 
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

# Image Preprocessing :
def get_img_array(img_file, size = (img_height, img_width), preprocess = True):
    
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize(size)
    img = np.array(img)
    print("Array Dims :",img.shape)
    
    # Pour prediction seulement : batch + VGG19preprocessing
    if preprocess == True:
        img = np.expand_dims(img, axis = 0)
        print("After Expand :", img.shape)
    
        img = preprocess_input(img)   
        print("After VGG19 PP :", img.shape)
    return img


# Main function
def load_vgg19(img_file):
    # Import model
    model = load_model(vgg19_path)
    
    # Preprocessing de l'image
    img = get_img_array(img_file, size = (img_height, img_width))
    img_orig = get_img_array(img_file, size = (img_height, img_width), preprocess = False)
    print("load_vgg19 : preprocess - ", "img : ", img.shape, "img_orig :", img_orig.shape)
    
    # Prediction :
    preds = model.predict(img)[0]
    sorted_indexes = np.flip(np.argsort(preds))
    sorted_preds = [preds[i] for i in sorted_indexes]
    sorted_classes = [classes[i] for i in sorted_indexes]
    
    print(preds)
    print(sorted_classes)

    # Grad-CAM : plot the 3 most probable classes :
    fig = plt.figure(figsize = (8,8))
    
    gcams = []
    for i, id in enumerate(sorted_indexes[:3]):    

        superimposed_img = gradcam(model, img, img_orig, 
                                   img_height, img_width, 
                                   class_index = id, alpha = 0.8)
        
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(superimposed_img)
        ax.set_title('Grad-CAM for '+ sorted_classes[i], fontsize = 14)
        ax.text(x = 10, y = 30, s = 'P(%s) = %0.3f'%(sorted_classes[i], sorted_preds[i]), fontsize = 14)
        
        plt.grid(None)
        plt.axis('off')
        
        gcams.append(superimposed_img)
        
    return fig, gcams