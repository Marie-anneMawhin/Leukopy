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
from  import_func_VGG16SVM import *









model_list = ["vgg16+SVM", "vgg19", "ViT"]


def main(): #write():
    
        st.subheader('Classification')

        select_model=st.selectbox("Select model", model_list)

 
        
        image_file=st.file_uploader("Upload image",type=['jpg','tiff','png'])  
        
        
        if image_file is not None:
           
            image=image_file.read()   
            
            
            file_details={"Filename":image_file.name,"File shape":image_file.size,"Filetype":image_file.type}           
            st.write(file_details)
            
            
            
            
            col1,col2=st.beta_columns(2)
            with col1:
                with st.beta_expander("Original Image"):
                    
                  st.image(image,width=150)
                  
                with st.beta_expander("True class"): 
                    
  
                        
                        True_class=Function_read_true_class(image_file)
                        if True_class==" ":
                            st.write("Not in data base")
                        else:
                            st.write(True_class)
                  
   
                    
            with col2:
                
   
                with st.beta_expander("Classified as"):
                ################### begin vgg16+SVM #########################                  
                    
                  if select_model=="vgg16+SVM":
                      
                     
                     Model_flag=Function_choose_model(image_file)                
                     
                     
                     if  Model_flag=="VGG16_SVM_6_C_SF_flag":
   
                              base_model, str_result,img=Function_VGG16_SVM_6_C_SF(image_file)
                              st.write(str_result) 
         
                                                          
                     if  Model_flag=="VGG16_SVM_6_C_AF_flag":
                           
                              base_model, str_result,img=Function_VGG16_SVM_6_C_AF(image_file)
                              st.write(str_result)                                                        
                                                                 
                    
                     if  Model_flag=="VGG16_SVM_8_C_AF_flag":
                         
     
                              base_model, str_result,img=Function_VGG16_SVM_8_C_AF(image_file)
                              st.write(str_result)
                                                         
                                                                                               
                     if  Model_flag=="VGG16_SVM_8_C_SF_flag":
                         
  
                              base_model, str_result,img=Function_VGG16_SVM_8_C_SF(image_file)
                              st.write(str_result)   
                ################### end vgg16+SVM #########################"                                  
                              
                with st.beta_expander("Grad-CAM"):
                    
                    
                    
                ################### begin vgg16+SVM #########################   
                    
                   if select_model=="vgg16+SVM":
                                     big_heatmap, superimposed_img = gradcam(base_model, img, alpha = 0.8, plot = False)  
                                     st.image(superimposed_img,width=150) 
                                                                                             
                       
                 ################### end vgg16+SVM #########################"                              
 
                              
  
   
    
    
    
if __name__=='__main__':
   main()