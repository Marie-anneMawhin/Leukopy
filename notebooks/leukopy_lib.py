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

def generate_dataframes(data_dir = None,
                        random_state = 42, 
                        random_lock = True):
    
    """ Generate 3 .csv dataframes (train, validation, test) from folder Leukopy/data (GitHub). 
    Image folders are grouped by cell types and images either label by :
    12 classes (label):  BA, BL, BNE, ERB, EO, LY, MMY, MO, MY, PLT, PMY, SNE
    9 classes (label2):  BA, BL, IG, ERB, EO, LY, MO, PLT, NEU
    10 classes (label3):  BA, BL, BNE, ERB, EO, LY, IG, MO, PLT, SNE
        
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
    data['cell_type'] = [image_path.parts[-2] for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]

    data['label'] = [image_path.stem.split('_')[0] for ext in ['jpg', 'tif', 'png'] for image_path in path.glob(f'**/*.{ext}')]
    data['label'] = data['label'].replace(to_replace = ["NEUTROPHIL", "PLATELET"], 
                                  value = ["NEU", "PLT"])
    
    data['label_2'] = data['label'].replace(to_replace = ["SNE","BNE","NEUTROPHIL", "MY","MMY","PMY", "PLATELET"], 
                                  value = ["NEU","IG","NEU", "IG","IG","IG", "PLT"])
    
    data['label_3'] = data['label'].replace(to_replace = ["NEUTROPHIL", "MY","MMY","PMY", "PLATELET"], 
                                  value = ["NEU", "IG","IG","IG", "PLT"])
    # Conversion to DataFrames
    df_train, df_test = train_test_split(data, test_size = 0.15, random_state = random_state)
    df_train, df_valid = train_test_split(df_train, test_size = 0.12, random_state = random_state)
    
    # Save DFs : .CSV files
    df_train.to_csv(path_or_buf = 'train_set.csv')
    df_valid.to_csv(path_or_buf = 'valid_set.csv')
    df_test.to_csv(path_or_buf = 'test_set.csv')
    
    return df_train, df_test, df_valid