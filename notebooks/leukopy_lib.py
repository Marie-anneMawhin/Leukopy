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