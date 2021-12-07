# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:26:25 2021

@author: laleh
"""
import streamlit as st
import pandas as pd
import numpy as np



from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
import sklearn
import scipy
import os
import csv
import pickle
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input,VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.svm import SVC
from PIL import Image,ImageOps

from tensorflow.keras.models import save_model, load_model
import tensorflow
import sys
import pathlib
import matplotlib.cm as cm
import tensorflow as tf






######################### begin Grad-CAM

def get_img_array(img):
  img_height, img_width=224,224
  #img = tf.keras.preprocessing.image.load_img(img_path, target_size = size)
  
  array = tf.keras.preprocessing.image.img_to_array(img)
  array = np.expand_dims(array, axis = 0)
  array=preprocess_input(array)
  return array

def make_heatmap(img_array, model1, last_conv_layer, class_index):

  grad_model = tf.keras.models.Model([model1.inputs], [last_conv_layer.output, model1.output])
  
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

def gradcam(model1, img, image_file,class_index = None, alpha = 0.5, plot = True):

  # Détecte la dernière couche de convolution (pas terrible : il faudrait sélectionner sur le type, pas sur le nom) :
  for layer in reversed(model1.layers):                ################
    if 'conv' in layer.name:
      last_conv_layer = model1.get_layer(layer.name)
      break
  
  # Chargement + mise en forme de l'image :
  img_array = get_img_array(img)
  
  
  
  
  #return np.array(img).shape,0
  if np.array(img_array).shape[3]==4:  #si c'est un fichier tiff
          im=Image.open(image_file)
          img=im.convert("RGB")
          img_height, img_width=224,224
         
          img_array = tf.keras.preprocessing.image.img_to_array(img)  
          img_array = np.expand_dims(img_array, axis = 0) 
          img_array=preprocess_input(img_array)                
                                #img_array = img_array.convert("RGB")
                                #img_array = img_array.resize((224, 224))   
  
  
  """
  # Choix de la classe à représenter :
  if class_index == None :
    # Désactiver Sotfmax sur la couche de sortie :
    model1.layers[-1].activation = None
    # Prédiction + classe la plus probable :
    predict = model1.predict(img_array)
    class_index = np.argmax(predict[0])
 """
  # Calcul de la CAM : resize pour comparaison avec l'image finale
  
  heatmap = make_heatmap(img_array, model1, last_conv_layer, class_index)
  big_heatmap = heatmap
  #big_heatmap = cv2.resize(heatmap, dsize = (img_height, img_width), interpolation = cv2.INTER_CUBIC)

  ## Traitement de la Heatmap
  # 1/ Normalisation
  big_heatmap = big_heatmap/big_heatmap.max()
  # 2/ On passe dans ReLu, pour flinguer les valeurs négatives
  big_heatmap = np.maximum(0, big_heatmap)
  
  ## Superposition de l'image et de la Heatmap 
  # 1/ Import de l'image d'origine
  ###############"img = tf.keras.preprocessing.image.load_img(img_path)
  ###############img = tf.keras.preprocessing.image.img_to_array(img)
 
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
    plt.title("Chosen class : "+str(list(label_map.keys())[class_index]))
  return big_heatmap, superimposed_img

######################### end Grad-CAM











def Function_choose_model(image_file):
    Mendeley=["PLATELET", "BNE","MO", "LY", "IG", "ERB", "EO","BA","SNE","MY","MMY","PMY"]

    text=image_file.name
    indicator=text.split('.')[0].split('_')[0]
    
    
    
    if (indicator in Mendeley) :
       if ("Filt" in image_file.name):      
          return "VGG16_SVM_6_C_AF_flag"
       else: 
          return "VGG16_SVM_6_C_SF_flag"  
      
    else:    #cas de AML, Raabin ou autre
      
   
       if ("Filt" in image_file.name):      
          return "VGG16_SVM_8_C_AF_flag"
       else: 
          return "VGG16_SVM_8_C_SF_flag"         
      
        
      
        
      
        
    

def Function_VGG16_SVM_6_C_SF(image_file):   
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/") 
                              P=pathlib.PureWindowsPath(path_L)
                              path_L=str(P.parents[0])+"/weights"                          

                              output_classes1=["BNE","EO","IG","PLT","ERB","MON","BAS","LY"]
                              image2=Image.open(image_file)
                              X_img=[]
                              size=(224,224) 
                              img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                              X_img.append(np.asarray(img))
                              X_test_img = np.array(X_img)
                    
                              
                              
                              model = load_model(path_L+"/MTL_MTL_Segment_8classes_Correspond_T")                        
      
                              
                              
                             
                              intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                              TT=preprocess_input(X_test_img)
                              X_test_features = intermediate_layer_model.predict(TT)
                             
                              grid = pickle.load(open(path_L+'/svm_MTL_Segment_8classes_Correspond_T.pkl','rb'))
                             

                              y_est1=grid.predict(X_test_features)
                              
                              if (output_classes1[int(y_est1)]=="IG")  or (output_classes1[int(y_est1)]=="BNE") :
                       
                                  output_classes2=["others","BNE_BNE","BNE_SNE"," IG_MY","IG_MMY","IG_PMY"]  
                                  image2=Image.open(image_file)
                                  X_img=[]
                                  size=(224,224) 
                                  img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                                  X_img.append(np.asarray(img))
                                  X_test_img = np.array(X_img)
                        
                                 
                                 
                                  model = load_model(path_L+"/model_Transfer_Learning_segment5_Correspond")
                                  intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                                  TT=preprocess_input(X_test_img)
                                  X_test_features = intermediate_layer_model.predict(TT)
                                 
    
                                  
                                  grid = pickle.load(open(path_L+'/svm_Transfer_Learning_segment5_Correspond.pkl','rb'))

    
                                  y_est2=grid.predict(X_test_features)
                                  
                                  if output_classes2[int(y_est2)]=="others":
                                                                       
                                      return model.layers[0], output_classes1[int(y_est1)],image2
                                  else:
                                      return model.layers[0], output_classes2[int(y_est2)],image2
                              else:
                                  return model.layers[0], output_classes1[int(y_est1)],image2
                                 
                                    
def Function_VGG16_SVM_6_C_AF(image_file): 
    
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/") 
                              P=pathlib.PureWindowsPath(path_L)
                              path_L=str(P.parents[0])+"/weights"                                
                                    
                              output_classes1=["BNE","EO","IG","PLT","ERB","MON","BAS","LY"]
                              image2=Image.open(image_file)
                              X_img=[]
                              size=(224,224) 
                              img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                              X_img.append(np.asarray(img))
                              X_test_img = np.array(X_img)
                    
                             
                             
                              model = load_model(path_L+"/MTL_MTL_Segment_8classes_T")
                              intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                              TT=preprocess_input(X_test_img)
                              X_test_features = intermediate_layer_model.predict(TT)
                             

                             
                              grid = pickle.load(open(path_L+'/svm_MTL_Segment_8classes_T.pkl','rb'))
                              y_est1=grid.predict(X_test_features)
                              
                              if (output_classes1[int(y_est1)]=="IG")  or (output_classes1[int(y_est1)]=="BNE") :                        
                         
                         
                                  output_classes2=["others","BNE_BNE","BNE_SNE"," IG_MY","IG_MMY","IG_PMY"] 
                                  image2=Image.open(image_file)
                                  X_img=[]
                                  size=(224,224) 
                                  img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                                  X_img.append(np.asarray(img))
                                  X_test_img = np.array(X_img)
                        
                                 
                                  
                                  model = load_model(path_L+"/model_Transfer_Learning_segment5")
                                  intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                                  TT=preprocess_input(X_test_img)
                                  X_test_features = intermediate_layer_model.predict(TT)
                                 
    
                                  
                                  grid = pickle.load(open(path_L+'/svm_Transfer_Learning_segment5.pkl','rb'))
    
                                  y_est2=grid.predict(X_test_features)
                                  
                                  if output_classes2[int(y_est2)]=="others":
                                                                       
                                      return model.layers[0], output_classes1[int(y_est1)],image2
                                  else:
                                      return model.layers[0], output_classes2[int(y_est2)],image2
                              else:
                                  return model.layers[0], output_classes1[int(y_est1)],image2
                              
def  Function_VGG16_SVM_8_C_AF(image_file):  
    
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/") 
                              P=pathlib.PureWindowsPath(path_L)
                              path_L=str(P.parents[0])+"/weights"                            
                              
                              output_classes=["BNE","EO","IG","PLT","ERB","MON","BAS","LY"]
                              image2=Image.open(image_file)
                              X_img=[]
                              size=(224,224) 
                              img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                              if np.array(img).shape[2]==4:  #si c'est un fichier tiff
                       
                                img = img.convert("RGB")
                                img = img.resize((224, 224))                          
                              
                              
                              X_img.append(np.asarray(img))
                              X_test_img = np.array(X_img)
                    
                             
                             
                              model = load_model(path_L+"/MTL_MTL_Segment_8classes_T")
                              intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                              TT=preprocess_input(X_test_img)
                              X_test_features = intermediate_layer_model.predict(TT)
                             

                              
                              grid = pickle.load(open(path_L+'/svm_MTL_Segment_8classes_T.pkl','rb'))

                              y_est=grid.predict(X_test_features)
                              return  model.layers[0], output_classes[int(y_est)],image2
                              
                          
def  Function_VGG16_SVM_8_C_SF(image_file):   
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/") 
                              P=pathlib.PureWindowsPath(path_L)
                              path_L=str(P.parents[0])+"/weights"                          
                              output_classes=["BNE","EO","IG","PLT","ERB","MON","BAS","LY"]
                              image2=Image.open(image_file)
                              X_img=[]
                              size=(224,224) 
                              img = ImageOps.fit(image2,size,Image.ANTIALIAS)
                              if np.array(img).shape[2]==4:  #si c'est un fichier tiff
                       
                                img = img.convert("RGB")
                                img = img.resize((224, 224))                              
                              
                              
                              
                              X_img.append(np.asarray(img))
                              X_test_img = np.array(X_img)
                    
                             
                             
                              model = load_model(path_L+"/MTL_MTL_Segment_8classes_Correspond_T")
                              intermediate_layer_model =Model( inputs=model.input, outputs=model.layers[2].output)
                              TT=preprocess_input(X_test_img)
                              X_test_features = intermediate_layer_model.predict(TT)
                             

                             
                              grid = pickle.load(open(path_L+'/svm_MTL_Segment_8classes_Correspond_T.pkl','rb'))

                              y_est=grid.predict(X_test_features)
                              return  model.layers[0], output_classes[int(y_est)],image2 
                                                            
                                
def  Function_read_true_class(image_file):
    
                        TT=pathlib.Path(__file__).parent.resolve()
                        path_L=str(TT).replace("\\","/") 
                        df=pd.read_csv(path_L+"/labels.csv") 
                        dft=df[df.name==image_file.name]
                        if len(dft):
                            return str(dft["label"].values[0])   
                        else:
                            return " "
                                
                              