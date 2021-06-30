import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io, color, exposure, transform, img_as_float32
import skimage

from sklearn.model_selection import train_test_split

from pathlib import Path

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

def load_df_tf_dir(path_name, selection_list=None):
    '''
    Load dataset from directory and perform train, test, split.
    
    Args:
    -path_name: path to file as str
    -selection_list: list of selected labels
    
    return tuple of df (train, valid, test)
    '''
    path = Path(path_name)
    df = pd.DataFrame()
    df['img_paths'] = [str(image) for image in path.glob('*/*')]
    df['label'] = [image.stem.split('_')[0] for image in path.glob('*/*')]
    
    if selection_list: df = df[df['label'].isin(selection_list)]
    
    df_temp, df_test = train_test_split(df, test_size=0.15, random_state=42)
    df_train, df_valid = train_test_split(df_temp, test_size=0.12, random_state=42)
    return df_train, df_valid, df_test