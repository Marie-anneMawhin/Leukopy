import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys, os, glob, shutil

from collections import Counter
from dask import bag, diagnostics


from skimage import io, color, exposure, transform, img_as_float32
import skimage

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


from pathlib import Path

import tensorflow as tf

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow import keras
import pickle


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, EarlyStopping

from tensorflow.keras.utils import Sequence
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
###################### General functions ######################

def load_image(filename: str,  as_grey: bool = False, rescale: float = None, float32: bool = True) -> np.ndarray:
    '''
    load images from path
    
    Args:
    - filename: str, path
    - as_grey: bool, choose where to import as grey
    - rescale: float, reshape image to a factor
    - float32: reduce the precision to 32 instead of 64
    
    return loaded iamge as np.array
    '''
    
    if as_grey:
        image = skimage.io.imread(filename, as_gray=True)
        image = transform.resize(image, (363, 360)) #resize outliers
        
    else:
        image = skimage.io.imread(filename)
        image = transform.resize(image, (363, 360, 3)) #resize outliers
        
        
    if rescale: image = transform.rescale(image, rescale, anti_aliasing=True) #reduce dim

    if float32: image = img_as_float32(image)  # Optional: set to your desired precision
    
    return image


def add_columns(filename):
    
    im = io.imread(filename)
    temp = pd.DataFrame(index=[0])

    temp['height'] = im.shape[0] 
    temp['width'] = im.shape[1]
    temp['mean_brightness'] = np.mean(im)    
    
    im_gray = color.rgb2grey(im)
    temp['mean_luminance'] = np.mean(im_gray)
    
    return temp

def generate_df_dask(path):
    df = pd.DataFrame()
    df['img_path'] = [str(image_path) for ext in ['jpg', 'tiff', 'png'] 
                      for image_path in path.glob(f'**/*.{ext}')]

    df['label'] = [image_path.parts[-2] for ext in ['jpg', 'tiff', 'png'] 
                   for image_path in path.glob(f'**/*.{ext}')]

    df['label_2'] = [image_path.stem.split('_')[0] 
                     for ext in ['jpg', 'tiff', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    
    barca = ['MO', 'ERB', 'PLATELET', 'BA', 'BNE', 'SNE', 'LY', 'EO', 'MMY', 'PMY', 'MY']
    munich = ['MON', 'EBO', 'BAS', 'NGB', 'NGS', 'LYT', 'EOS', 'MMZ', 'PMO', 'MYB']

    df.loc[df.label_2.isin(barca),'origin']='barcelone'
    df.loc[df.label_2.isin(munich),'origin']='munich'
    df.origin.fillna('raabin', inplace=True)
    
    addcol_bag = bag.from_sequence(df.img_path.to_list()).map(add_columns)
    with diagnostics.ProgressBar():
        res = addcol_bag.compute()
        
    res_df = pd.concat(res).reset_index(drop=True)
    df_temp = df.join(res_df)
    return df_temp

def load_df(path_name, from_folder=False):
    '''
    Args:
    -path_name: path to file as str
    -from_folder: True to generate df from image folder.
    '''
    path = Path(path_name)
    
    if from_folder:
        df = generate_df_dask(path)
        
    else:
        df = pd.read_csv(path)
    
    return df


def create_folders(df, name, col):
    dest = Path(f'../../data/main_dataset/{name}')
    dest.mkdir(parents=True, exist_ok=True)
    
    for f in df[col]:
        shutil.copy(f, dest)
        
def generate_dataframes(data_dir = None,
                        random_state = 42, 
                        random_lock = True):
    
    """ Generate 3 .csv dataframes (train, validation, test) from folder Leukopy/data (GitHub). 
       
    data_dir : Path Object or str, path to images
    random_state : int, default 42 for reproducibility
    random_lock : bool., lock the value of random_state
    """
    # To work with the same split. 
    if random_lock == True:
        if random_state != 42:
            random_state = 42
            
    # Verification : path to data files
    if data_dir == None:
        return "Please provide the path of the directory 'Leukopy' "
    
    path = Path(data_dir)
  
    
    data = pd.DataFrame()
    data['img_paths'] = [image_path for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]
       
    #for sanity check
    data['label'] = [image_path.stem.split('_')[0] for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['label'] = data['label'].replace(to_replace = ["NEUTROPHIL", "PLATELET"], 
                                  value = ["SNE", "PLT"])
    
    #remove label IG
    data = data[~(data.label == 'IG')] 

    # Conversion to DataFrames
    df_train, df_test = train_test_split(data, test_size = 0.15, random_state = random_state)
    df_train, df_valid = train_test_split(df_train, test_size = 0.12, random_state = random_state)
    
    # Save DFs : .CSV files
    df_train.to_csv(path_or_buf = path/'train_set.csv')
    df_valid.to_csv(path_or_buf = path/'valid_set.csv')
    df_test.to_csv(path_or_buf = path/'test_set.csv')
    
    
    return df_train, df_test, df_valid



def generate_images_df(data_dir = None):
    
    """ Generate 3 .csv dataframes (train, validation, test) from folder Leukopy/data (GitHub). 
    Image folders are grouped by cell types and images either label by :
    12 classes (label):  BA, BL, BNE, ERB, EO, LY, MMY, MO, MY, PLT, PMY, SNE
    9 classes (label2):  BA, BL, IG, ERB, EO, LY, MO, PLT, NEU
    10 classes (label3):  BA, BL, BNE, ERB, EO, LY, IG, MO, PLT, SNE
        
    data_dir : Path Object or str, path to images

    """
   
    # Verification : path to data files
    if data_dir == None:
        return "Please provide the path of the directory 'Leukopy' "
    
    path = Path(data_dir)
  
    data = pd.DataFrame()
    data['img_path'] = [str(image_path) for ext in ['jpg', 'tiff', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['cell_type'] = [image_path.parts[-2] for ext in ['jpg', 'tiff', 'png'] for image_path in path.glob(f'**/*.{ext}')]

    data['label'] = [image_path.stem.split('_')[0] for ext in ['jpg', 'tiff', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['label'] = data['label'].replace(to_replace = ["NEUTROPHIL", "PLATELET"], 
                                  value = ["SNE", "PLT"])
    #remove label IG if needed
    data = data[~(data.label == 'IG')].reset_index(drop=True)
    
    data['label_2'] = data['label'].replace(to_replace = ["BNE","NEUTROPHIL", "MY","MMY","PMY", "PLATELET"], 
                                  value = ["IG","SNE", "IG","IG","IG", "PLT"])
    
    data['label_3'] = data['label'].replace(to_replace = ["NEUTROPHIL", "MY","MMY","PMY", "PLATELET"], 
                                  value = ["SNE", "IG","IG","IG", "PLT"])
    
    
    return data

############################################## DL FUNCTIONS ##############################################
def sel_n_classes(df, n_classes):
    
    # 11 classes, without BL
    if n_classes == 11:
        df = df[df["label"] != "BL"]
      
  # 8 classes (IG, SNE), without BL :
    if n_classes == 8:
        df = df[df["label"] != "BL"]
        
        df["label"] = df["label_2"]
       
  # 9 classes (IG, SNE, BL)
    if n_classes == 9:
        df["label"] = df["label_2"]

    return df

def choose_classes(df_train, df_test, df_valid, n_classes=12):
    return n_classes, sel_n_classes(df_train, n_classes), sel_n_classes(df_test, n_classes), sel_n_classes(df_valid, n_classes)


# Calcul des poids pour compenser le déséquilibre des classes

def compute_weights(training_set, method = None):

    if method == 1:
        counter = Counter(training_set.classes)     
        class_weights = {class_id : float(max(counter.values()))/num_images for class_id, num_images in counter.items()} 

    if method == 2:
        class_weights = compute_class_weight(class_weight = 'balanced',
                                             classes = np.unique(training_set.classes),
                                             y = training_set.classes)
        class_weights = dict(enumerate(class_weights))

    else:
        counter = Counter(training_set.classes)                          
        class_weights = {class_id : (1/num_images)*float(sum(counter.values()))/2 for class_id, num_images in counter.items()} 
    return class_weights



############################################## GRADCAM VIT ##############################################
def get_img_array(img_path: str, dim: tuple) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(Path(img_path), target_size = dim)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(img, axis = 0)
    return array, img

def make_gradcam_heatmap(img_array: np.ndarray, model, layer_name: str):
    
    for layer in reversed(model.layers):
        if layer_name in layer.name:
            last_conv_layer = model.get_layer(layer.name)
            break

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output,  model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    last_conv_layer_output = last_conv_layer_output#[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)[1:]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam(img_path: str, img_size: int, model: Model, layer_name: str) -> 'image':
    
    arr, img = get_img_array(img_path, (img_size, img_size))
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = make_gradcam_heatmap(arr, model, layer_name)
    heatmap = np.uint8(255 * heatmap)
    
    size_embedding = model.layers[1].output_shape[1]
    heatmap = np.reshape(heatmap, (size_embedding,size_embedding))
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[..., :-1]
    jet_heatmap = jet_colors[heatmap]


    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.8 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

    
    
# # Specific to colab
# # Gives access to the Drive
# from google.colab import drive
# import tensorflow as tf
# import sys, os


# drive.mount('/content/drive')

# # GPU status verification
# tf.test.gpu_device_name()

# # GPU type verification
# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#     print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
#     print('and then re-execute this cell.')
# else:
#     print(gpu_info)

# # Need to copy all the files on the local computer
# !cp -r "drive/MyDrive/data/main_dataset.zip" .
# !unzip main_dataset.zip

# sys.path.append('drive/MyDrive/colab_notebooks/')