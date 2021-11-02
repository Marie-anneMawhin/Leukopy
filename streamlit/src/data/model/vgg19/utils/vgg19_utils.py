import streamlit as st

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image


## PARAMETERS

img_height = 360
img_width  = 360
classes = ["BA","BNE","EO","ERB","LY","MMY","MO","MY","PLT","PMY","SNE"]
label_map = {'BA': 0, 'BNE': 1, 'EO': 2, 'ERB': 3, 'LY': 4, 'MMY': 5, 'MO': 6, 'MY': 7, 'PLT': 8, 'PMY': 9, 'SNE': 10}


def load_model(path):
    """
    Load a model whose weights are saved in Leukopy/streamlit/src/data/model/vgg19/weights

    Parameters
    ----------
    path : Path object, 
        Chemin d'accès conduisant aux poids du modèle.

    Returns
    -------
    model : tf.Model()
        Modèle pré-entraîné (poids, structure, optimiseur).

    """

    model = tf.keras.models.load_model(path)
    return model


def get_img_array(img_file, size = (img_height, img_width)):
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize(size)
    img = np.array(img)
    print("Array Dims :",img.shape)
    
    img = np.expand_dims(img, axis = 0)
    print("After Expand :", img.shape)
    img = preprocess_input(img)   
    print("After VGG19 PP :", img.shape)
    return img


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


def gradcam(model, img_path, img_height, img_width, class_index = None, alpha = 0.5, plot = True):
    
    # Désactive softmax sur la dernière couche :
    model.layers[-1].activation = None
    
    # Détecte la dernière couche de convolution du modèle :
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = model.get_layer(layer.name)
            break

    # Chargement + preprocessing de l'image :
    img_array = get_img_array(img_path, size = (img_height, img_width))
    
    # Choix de la classe à représenter (si class_index non renseigné) :
    if class_index == None :
        # Trouve la classe la plus probable :
        predict = model.predict(img_array)
        class_index = np.argmax(predict[0])

    # Calcul de la CAM : resize pour superposition avec l'image finale
    heatmap = make_heatmap(img_array, model, last_conv_layer, class_index)
    big_heatmap = heatmap
    
    # Réactive softmax :
    model.layers[-1].activation = tf.keras.activations.softmax

    ## Traitement de la Heatmap
    # Applique ReLu (élimine les valeurs négatives de la heatmap)
    big_heatmap = np.maximum(0, big_heatmap)
    # Normalisation
    big_heatmap = big_heatmap/big_heatmap.max()
    
    ## Superposition de l'image et de la heatmap 
    # 1/ Import de l'image d'origine
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # 2/ Rescale heatmap: 0-255
    big_heatmap = np.uint8(255*big_heatmap)
    # 3/ Jet colormap
    jet = cm.get_cmap("jet")
    # 4/ Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[big_heatmap]
    # 5/ Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    # 6/ Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    if plot == True:
        # 7/ Affichage des résultats
        fig = plt.figure(figsize = (8,8))
        fig.add_subplot(1,2,1)
        plt.imshow(big_heatmap)

        fig.add_subplot(1,2,2)
        plt.imshow(superimposed_img)
        plt.title("Chosen class : "+str(list(label_map.keys())[class_index]))

    return big_heatmap, superimposed_img


def full_prediction(model, img_file, size = (img_height, img_width), label_provided = False):

    print("In full_prediction !")
    
    # Preprocessing :
    if img_file is not None:
        img = get_img_array(img_file, size)
        
    print("Preprocessing done !")

    # Prediction
    probas = model.predict(img)[0]
    sorted_indexes = np.flip(np.argsort(probas))
    sorted_classes = [classes[i] for i in sorted_indexes]
    sorted_probas = [probas[i] for i in sorted_indexes]
    
    print(sorted_classes[0], sorted_probas[0])

    # Plot (3 classes les plus probables)
    #fig = plt.figure(figsize = (12,12))
    #ax1 = fig.add_subplot(1,1,1)
    #st.image(img, width=150)
    #ax1.text(x = 10, y = 25, s = 'P(%s) = %0.3f'%(sorted_classes[0], sorted_probas[0]), fontsize = 'xx-large')
    #ax1.text(x = 10, y = 55, s = 'P(%s) = %0.3f'%(sorted_classes[1], sorted_probas[1]), fontsize = 'xx-large')
    #ax1.text(x = 10, y = 85, s = 'P(%s) = %0.3f'%(sorted_classes[2], sorted_probas[2]), fontsize = 'xx-large')
    #plt.grid(None)
    #plt.axis('off')

    #ax2 = fig.add_subplot(1,4,2)
    #big_heatmap, superimposed_img = gradcam(model, img_path, img_height, img_width, 
    #                                        class_index = sorted_indexes[0], alpha = 0.8, plot = False)
    #ax2.imshow(superimposed_img)
    #ax2.set_title('Grad-CAM for '+sorted_classes[0], fontsize = 24)
    #plt.grid(None)
    #plt.axis('off')

    #ax3 = fig.add_subplot(1,4,3)
    #big_heatmap, superimposed_img = gradcam(model, img_path, img_height, img_width, 
    #                                        class_index = sorted_indexes[1], alpha = 0.8, plot = False)
    #ax3.imshow(superimposed_img)
    #ax3.set_title('Grad-CAM for '+sorted_classes[1], fontsize = 24)
    #plt.grid(None)
    #plt.axis('off')

    #ax4 = fig.add_subplot(1,4,4)
    #big_heatmap, superimposed_img = gradcam(model, img_path, img_height, img_width, 
    #                                        class_index = sorted_indexes[2], alpha = 0.8, plot = False)
    #ax4.imshow(superimposed_img)
    #ax4.set_title('Grad-CAM for '+sorted_classes[2], fontsize = 24)
    #plt.grid(None)
    #plt.axis('off')

    return
