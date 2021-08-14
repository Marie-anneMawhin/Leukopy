import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import Counter


from skimage import io, color, exposure, transform, img_as_float32
import skimage

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


from pathlib import Path

import tensorflow as tf


###################### General functions ######################

def load_image(filename,  as_grey=False, rescale=None, float32=True):
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


def load_df(path_name):
    '''
    Args:
    -path_name: path to file as str
    '''
    path = Path(path_name)
    df = pd.read_csv(path_name)
    return df


def create_folders(df, name, col):
    dest = Path(f'../../data/main_dataset/{name}')
    dest.mkdir(parents=True, exist_ok=True)
    
    for f in df[col]:
        shutil.copy(f, dest)

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
    data['img_path'] = [str(image_path) for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['cell_type'] = [image_path.parts[-2] for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]

    data['label'] = [image_path.stem.split('_')[0] for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['label'] = data['label'].replace(to_replace = ["NEUTROPHIL", "PLATELET"], 
                                  value = ["SNE", "PLT"])
    
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



############################################## GRADCAM FUNCTIONS ##############################################

def get_img_array(img_path, img_size):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size = img_size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis = 0)
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

def gradcam(model, img_path, class_index = None, alpha = 0.5, plot = True):

    # Détecte la dernière couche de convolution (pas terrible : il faudrait sélectionner sur le type, pas sur le nom) :
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
          last_conv_layer = model.get_layer(layer.name)
          break

    # Chargement + mise en forme de l'image :
    img_array = get_img_array(img_path, size = (img_height, img_width))

    # Choix de la classe à représenter :
    if class_index == None :
    # Désactiver Sotfmax sur la couche de sortie :
        model.layers[-1].activation = None
        # Prédiction + classe la plus probable :
        predict = model.predict(img_array)
        class_index = np.argmax(predict[0])

    # Calcul de la CAM : resize pour comparaison avec l'image finale
    heatmap = make_heatmap(img_array, model, last_conv_layer, class_index)
    big_heatmap = heatmap
    #big_heatmap = cv2.resize(heatmap, dsize = (img_height, img_width), interpolation = cv2.INTER_CUBIC)

    ## Traitement de la Heatmap
    # 1/ Normalisation
    big_heatmap = big_heatmap/big_heatmap.max()
    # 2/ On passe dans ReLu, pour flinguer les valeurs négatives
    big_heatmap = np.maximum(0, big_heatmap)

    ## Superposition de l'image et de la Heatmap 
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