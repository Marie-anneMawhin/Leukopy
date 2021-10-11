# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:16:17 2021

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











def main():
    
 
    st.title("Classification  App")
    
    menu=["Classification", "Filtering"]
    choice=st.sidebar.selectbox("Menu",menu)
    
    if choice=="Classification":
        st.subheader("Classification")
        
        image_file=st.file_uploader("Upload image",type=['jpg','tiff'])         
        VGG16_SVM_6_C_SF=st.sidebar.checkbox("VGG16+SVM avec subclasses sans filtre (data: Mendely Correspond)")
        VGG16_SVM_6_C_AF=st.sidebar.checkbox("VGG16+SVM avec subclasses avec filtre (data: Mendely)")
        VGG16_SVM_8_C_SF=st.sidebar.checkbox("VGG16+SVM sans subclasses sans filtre (data: 3 bases Correspond)")
        VGG16_SVM_8_C_AF=st.sidebar.checkbox("VGG16+SVM sans subclasses avec filtre (data: 3 bases)")
        

        
        if image_file is not None:
           
            image=image_file.read()   
            
            file_details={"Filename":image_file.name,"File shape":image_file.size,"Filetype":image_file.type}           
            st.write(file_details)
            
            
            
            
            col1,col2=st.beta_columns(2)
            with col1:
                with st.beta_expander("Original Image"):
                    
                  st.image(image,width=150)
                    
            with col2:
                with st.beta_expander("Classified as"):
                     if  VGG16_SVM_6_C_SF:
                         
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/")
                         
                              output_classes=["neutrophil","eosinophil","ig","platelet","erythroblast","monocyte","basophil","lymphocyte"]
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

                              y_est=grid.predict(X_test_features)
                              
                              if (output_classes[int(y_est)]=="ig")  or (output_classes[int(y_est)]=="neutrophil") :
                       
                                  output_classes=["others","neutrophil_BNE","neutrophil_SNE"," ig_MY","ig_MMY","ig_PMY"]  
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
    
                                  y_est=grid.predict(X_test_features)
                                  st.write(output_classes[int(y_est)])
                              else:
                                  st.write(output_classes[int(y_est)])
                                 

                     if  VGG16_SVM_6_C_AF:
                         
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/")
                         
                              output_classes=["neutrophil","eosinophil","ig","platelet","erythroblast","monocyte","basophil","lymphocyte"]
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

                              y_est=grid.predict(X_test_features)
                              
                              if (output_classes[int(y_est)]=="ig")  or (output_classes[int(y_est)]=="neutrophil") :                        
                         
                         
                                  output_classes=["others","neutrophil_BNE","neutrophil_SNE"," ig_MY","ig_MMY","ig_PMY"] 
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
    
                                  y_est=grid.predict(X_test_features)
                              
                                  st.write(output_classes[int(y_est)])
                              else:
                                  st.write(output_classes[int(y_est)])
                       
                                           
                    
                     if  VGG16_SVM_8_C_AF:
                         
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/")
                              output_classes=["neutrophil","eosinophil","ig","platelet","erythroblast","monocyte","basophil","lymphocyte"]
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
                              st.write(output_classes[int(y_est)])   
                              
                              
                     if  VGG16_SVM_8_C_SF:
                         
                              TT=pathlib.Path(__file__).parent.resolve()
                              path_L=str(TT).replace("\\","/")
                              output_classes=["neutrophil","eosinophil","ig","platelet","erythroblast","monocyte","basophil","lymphocyte"]
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
                              st.write(output_classes[int(y_est)])   
                                                            
                              
                              
                              
                              
                              
                              
                    
                   
                    #if normalize_case:
                     #   raw_text=raw_text.lower()
                    #st.write(raw_text)
                    #text_downloader(raw_text)
    else:
       st.subheader("Filtering")
    
    
    
if __name__=='__main__':
    main()