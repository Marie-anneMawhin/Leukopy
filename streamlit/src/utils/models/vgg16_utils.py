import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import pickle

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image, ImageOps

###############################################################################
# GRADCAM

def get_img_array(img):
  array = tf.keras.preprocessing.image.img_to_array(img)
  array = np.expand_dims(array, axis = 0)
  array = preprocess_input(array)
  return array

def make_heatmap(img_array, model, last_conv_layer, class_index):
  grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
  
  with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    class_channel = preds[:, class_index]
  grads = tape.gradient(class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))

  heatmap_tmp = last_conv_layer_output[0].numpy()

  for i in range(512):
    heatmap_tmp[:,:,i] *= pooled_grads[i]
  heatmap = np.mean(heatmap_tmp, axis=-1)
  return heatmap

def gradcam(model, img, image_file,class_index = None, alpha = 0.5, plot = True):

  for layer in reversed(model.layers):
    if 'conv' in layer.name:
      last_conv_layer = model.get_layer(layer.name)
      break
  # Chargement + mise en forme de l'image :
  img_array = get_img_array(img)
  
  if np.array(img_array).shape[3]==4:  #si c'est un fichier tiff
          im = Image.open(image_file)
          img = im.convert("RGB")
         
          img_array = tf.keras.preprocessing.image.img_to_array(img)  
          img_array = np.expand_dims(img_array, axis = 0) 
          img_array=preprocess_input(img_array)                

  """
  # Choix de la classe à représenter :
  if class_index == None :
    # Désactiver Sotfmax sur la couche de sortie :
    model.layers[-1].activation = None
    # Prédiction + classe la plus probable :
    predict = model.predict(img_array)
    class_index = np.argmax(predict[0])
    print("zorgbluff")
 """
  # Calcul de la CAM : resize pour comparaison avec l'image finale
  
  heatmap = make_heatmap(img_array, model, last_conv_layer, class_index)
  big_heatmap = heatmap

  ## Traitement de la Heatmap
  # 1/ Normalisation
  big_heatmap = big_heatmap/big_heatmap.max()
  # 2/ On passe dans ReLu, pour flinguer les valeurs négatives
  big_heatmap = np.maximum(0, big_heatmap)
  
  ## Superposition de l'image et de la Heatmap 
  # 2/ Rescale heatmap: 0-255
  big_heatmap = np.uint8(255*big_heatmap)

  # 3/ Jet colormap
  jet = cm.get_cmap("jet")

  # 4/ Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[big_heatmap]
  
  # 5/ Create an image with RGB colorized heatmap
  jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((np.array(img).shape[1], np.array(img).shape[0]))
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
  return big_heatmap, superimposed_img

###############################################################################

def choose_model(image_file):
    mendeley = ["PLATELET", "BNE","MO", "LY", "IG", "ERB", "EO","BA","SNE","MY","MMY","PMY"]

    indicator = image_file.name.split('.')[0].split('_')[0]
    
    if("Filt" in image_file.name):
        return "VGG16_SVM_6_C_AF_flag"
    else:
        return "VGG16_SVM_6_C_SF_flag"  
    
#    if (indicator in mendeley):
#       if ("Filt" in image_file.name):
#           return "VGG16_SVM_6_C_AF_flag"
#       else: 
#           return "VGG16_SVM_6_C_SF_flag"  
#      
#    else:    #cas de AML, Raabin ou autre
#       if ("Filt" in image_file.name):     
#           return "VGG16_SVM_8_C_AF_flag"
#       else: 
#           return "VGG16_SVM_8_C_SF_flag"         
  
###############################################################################

def VGG16_SVM_6_C_SF(image_file):
    
    classes = ["NEU","EO","IG","PLT","ERB","MO","BA","LY"]
    subclasses = ["others","BNE","SNE","MY","MMY","PMY"]
    vgg16_path = Path("./data/model/vgg16svm/MTL_MTL_Segment_8classes_Correspond_T")
    svm_path = Path("./data/model/vgg16svm/svm_MTL_Segment_8classes_Correspond_T.pkl")

    X_img = []
    image = Image.open(image_file)
    img = ImageOps.fit(image,(224,224),Image.ANTIALIAS)
    # Si c'est un fichier tiff
    if np.array(img).shape[2] == 4:
        img = img.convert("RGB")
        img = img.resize((224, 224))                              
    X_img.append(np.asarray(img))
    X_test_img = np.array(X_img)

    # Load VGG16 weights :
    model = load_model(vgg16_path)
    # Load SVM weights :
    with open(svm_path, 'rb') as f:
        grid = pickle.load(f)
 
    
    # Build main classes model
    intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
    # Feature extraction
    X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))  
    # SVM prediction
    class_label = grid.predict(X_test_features)

    # Subclasses model, if required :
    if (classes[int(class_label)] == "IG") or (classes[int(class_label)] == "NEU"):
        
        vgg16_path = Path("./data/model/vgg16svm/model_Transfer_Learning_segment5_Correspond")
        svm_path = Path("./data/model/vgg16svm/svm_Transfer_Learning_segment5_Correspond.pkl")
    
        # Load model : cnn + svm
        model = load_model(vgg16_path)
        with open(svm_path,'rb') as f:
            grid = pickle.load(f)
        
        # Feature extraction :
        intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
        X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))
        subclass_label = grid.predict(X_test_features)
                                  
        if subclasses[int(subclass_label)] == "others":
            return model.layers[0], classes[int(class_label)], image
        else:
            return model.layers[0], subclasses[int(subclass_label)], image
    else:
        return model.layers[0], classes[int(class_label)], image
            
###############################################################################  
                                 
def VGG16_SVM_6_C_AF(image_file): 

    classes = ["NEU","EO","IG","PLT","ERB","MO","BA","LY"]
    subclasses = ["others","BNE","SNE","MY","MMY","PMY"]
    vgg16_path = Path("./data/model/vgg16svm/MTL_MTL_Segment_8classes_T")
    svm_path = Path("./data/model/vgg16svm/svm_MTL_Segment_8classes_T.pkl")
         
    X_img = []
    image = Image.open(image_file)
    img = ImageOps.fit(image,(224,224),Image.ANTIALIAS)
    # Si c'est un fichier tiff
    if np.array(img).shape[2] == 4:
        img = img.convert("RGB")
        img = img.resize((224, 224))                              
    X_img.append(np.asarray(img))
    X_test_img = np.array(X_img)
           
    # Load VGG16 weights :           
    model = load_model(vgg16_path)
    # Load SVM weights :
    with open(svm_path,'rb') as f:
        grid = pickle.load(f)
    
    
    # Main classes model
    intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
    X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))
    class_label = grid.predict(X_test_features)
                        
    # Subclasses model, if required :      
    if (classes[int(class_label)] == "IG") or (classes[int(class_label)] == "NEU"):
        vgg16_path = Path("./data/model/vgg16svm/model_Transfer_Learning_segment5")
        svm_path = Path("./data/model/vgg16svm/svm_Transfer_Learning_segment5.pkl")
        
        # Load model : cnn + svm
        model = load_model(vgg16_path)
        with open(svm_path,'rb') as f:
            grid = pickle.load(f)
            
        # Feature extraction :
        intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
        X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))
        subclass_label = grid.predict(X_test_features)
        
        if subclasses[int(subclass_label)] == "others":
            return model.layers[0], classes[int(class_label)], image
        else:
            return model.layers[0], subclasses[int(subclass_label)], image
    else:
        return model.layers[0], classes[int(class_label)], image
        
###############################################################################
                      
def VGG16_SVM_8_C_AF(image_file):
    
    classes = ["NEU","EO","IG","PLT","ERB","MO","BA","LY"]
    vgg16_path = Path("./data/model/vgg16svm/MTL_MTL_Segment_8classes_T")
    svm_path = Path("./data/model/vgg16svm/svm_MTL_Segment_8classes_T.pkl")
    
    X_img = []
    image = Image.open(image_file)
    img = ImageOps.fit(image,(224,224),Image.ANTIALIAS)
    
    if np.array(img).shape[2] == 4:  #si c'est un fichier tiff
        img = img.convert("RGB")
        img = img.resize((224, 224))
    X_img.append(np.asarray(img))
    X_test_img = np.array(X_img)
                                           
    model = load_model(vgg16_path)
    with open(svm_path,'rb') as f:
        grid = pickle.load(f) 
    
    intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
    X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))

    class_label = grid.predict(X_test_features)
    
    return  model.layers[0], classes[int(class_label)], image
                              
###############################################################################
                            
def VGG16_SVM_8_C_SF(image_file):   
                                            
    classes = ["NEU","EO","IG","PLT","ERB","MO","BA","LY"]
    vgg16_path = Path("./data/model/vgg16svm/MTL_MTL_Segment_8classes_Correspond_T")
    svm_path = Path("./data/model/vgg16svm/svm_MTL_Segment_8classes_Correspond_T.pkl")    
    
    X_img = []
    image = Image.open(image_file)
    img = ImageOps.fit(image,(224,224),Image.ANTIALIAS)
    
    if np.array(img).shape[2]==4:  #si c'est un fichier tiff
        img = img.convert("RGB")
        img = img.resize((224, 224))                              
    X_img.append(np.asarray(img))
    X_test_img = np.array(X_img)
                    
    model = load_model(vgg16_path)
    with open(svm_path,'rb') as f:
        grid = pickle.load(f)     

    intermediate_layer_model = Model(inputs = model.input, outputs = model.layers[2].output)
    X_test_features = intermediate_layer_model.predict(preprocess_input(X_test_img))
                             
    class_label = grid.predict(X_test_features)

    return model.layers[0], classes[int(class_label)], image 