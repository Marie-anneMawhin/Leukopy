import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, color, exposure, transform, img_as_float32
import skimage

from scipy import stats

from pathlib import Path
import os, sys

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



def plot_random_by_classes(df, origin=False):
    label_list = list(df["label"].unique())
    
    fig = plt.figure(figsize = (20,10))
    for label, fi in zip (label_list, range(1,9)) :
        img_path = np.random.choice(df[df["label"] == label]["img_path"])
        img = load_image(img_path)
        plt.subplot(2,4,fi)
        plt.imshow(img)
        
        if origin:
            plt.title(f'{label} - {df[df["img_path"] == img_path]["origin"].values[0]}')
        else:
            plt.title(label+' - Height = '+str(img.shape[0])+' ; Width = '+str(img.shape[1]))
    return