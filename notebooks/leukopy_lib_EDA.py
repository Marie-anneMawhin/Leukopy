import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, color, exposure, transform, img_as_float32
import skimage

from scipy import stats

from pathlib import Path
import os, sys

from dask import bag, diagnostics

import leukopy_lib as leuko

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, LabelSet
output_notebook()

def plot_random_by_classes(df, origin=False):
    label_list = list(df["label"].unique())
    
    fig = plt.figure(figsize = (20,10))
    for label, fi in zip (label_list, range(1,9)) :
        img_path = np.random.choice(df[df["label"] == label]["img_path"])
        img = leuko.load_image(img_path)
        plt.subplot(2,4,fi)
        plt.imshow(img)
        
        if origin:
            plt.title(f'{label} - {df[df["img_path"] == img_path]["origin"].values[0]}')
        else:
            plt.title(label+' - Height = '+str(img.shape[0])+' ; Width = '+str(img.shape[1]))
    return

### Picture shape stats (all)
def plot_shape_stats(df):
    """
    Plot picture shape distribution
    """
    df_shape = pd.DataFrame(df[["height","width"]].value_counts(), columns = ["count"]).reset_index()
    df_shape["norm_count"] = np.sqrt(df_shape["count"])
    df_shape["norm_count"] = np.where(df_shape["count"] <= 10,2,df_shape["norm_count"])
    cds_df = ColumnDataSource(data = df_shape)  
    
    fig = figure(title = "Distribution de la taille des images en pixels",
                  x_axis_label = 'width (px)', y_axis_label = 'height (px)')
    fig.xaxis.minor_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None

    r1 = fig.circle(x = 'width', y = 'height', size = 'norm_count', source = cds_df,
                     hover_color = 'red', hover_alpha = 0.7)

    hv1 = HoverTool(renderers = [r1], tooltips = [('width','@width'),
                                                  ('height','@height'),
                                                  ('count','@count')])
    fig.add_tools(hv1)
    show(fig)
    return


### Mean brightness stats (cell type)
def plot_meanluminance_stats(df):
    luminance = pd.DataFrame(df.groupby(by = "label")["mean_luminance"].mean())
    luminance["class_std"] = df.groupby(by = "label")["mean_luminance"].std()
    luminance = luminance.rename({"mean_luminance":"class_mean"}, axis = 1)
    
    fig = plt.figure(figsize = (16,16))
    gs = fig.add_gridspec(8, 8, wspace = 0.6, hspace = 0.9)
    
    ### Boxplot
    f_ax1 = fig.add_subplot(gs[0:4,0:4])
    sns.boxplot(x = "label", y = "mean_luminance", data = df)
    f_ax1.set_title('Distribution de luminance pour les cellules de chaque classe (Boxplot)')
    
    ### Kdeplot
    f_ax2 = fig.add_subplot(gs[0:4,4:])
    sns.kdeplot(x = "mean_luminance", hue = "label", data = df)
    f_ax2.tick_params(labelbottom = True, labelleft = False, bottom = True, left = False)
    f_ax2.set_title('Distribution de luminance, classe par classe (KDEplot)')
    
    ### Errorbar
    f_ax3 = fig.add_subplot(gs[4:8, 0:4])
    plt.errorbar(x = luminance.index, y = luminance["class_mean"], 
                 yerr = luminance["class_std"],
                 marker = 'x', elinewidth = 1, linewidth = 0, ecolor = 'blue', color = 'red')
    plt.hlines(y = luminance["class_mean"].mean(), xmin = 'BA', xmax = 'PLT',
               color = "red", label = 'Mean Luminance - all classes')
    plt.xlabel('Classes')
    plt.ylabel('Mean Luminance / Class')
    f_ax3.set_title('Luminance moyenne par classe')
    
    ### Cells
    label_list = list(df["label"].unique())
    for label, fi in zip (label_list[0:4], range(4,8)):
        f_ax = fig.add_subplot(gs[5, fi])
        
        img = leuko.load_image(np.random.choice(df[df["label"] == label]["img_path"]))
        plt.imshow(img)
        
        f_ax.tick_params(labelbottom = False, labeltop = False,
                          labelleft = False, labelright = False,
                          bottom = False, top = False,
                          left = False, right = False)
        f_ax.set_title(label)
    
    for label, fi in zip (label_list[4:8], range(4,8)):
        f_ax = fig.add_subplot(gs[6, fi])
        
        img = leuko.load_image(np.random.choice(df[df["label"] == label]["img_path"]))
        plt.imshow(img)
        
        f_ax.tick_params(labelbottom = False, labeltop = False,
                          labelleft = False, labelright = False,
                          bottom = False, top = False,
                          left = False, right = False)
        f_ax.set_title(label)
    return


### exposure histogram (cell)
def plot_luminance_cell(filename, label = None, idx = None, histos = True):
    
    img = leuko.load_image(filename, as_grey = False)
    
    # Luminance
    lum = np.mean(img)
        
    if histos == True:
        # Canaux + gris
        reds = img[:,:,0]
        greens = img[:,:,1]
        blues = img[:,:,2]
        greys = img.mean(axis = 2)
    
        # Histogrammes
        histo = {}
        histo["red"] = exposure.histogram(reds)
        histo["green"] = exposure.histogram(greens)
        histo["blue"] = exposure.histogram(blues)
        histo["grey"] = exposure.histogram(greys)
    
        # Figure
        fig = plt.figure(figsize = (10,5))

        plt.subplot(1,2,1)
        plt.imshow(img)

        plt.subplot(1,2,2)
        plt.plot(histo["red"][1], histo["red"][0], color = "red")
        plt.plot(histo["green"][1], histo["green"][0], color = 'green')
        plt.plot(histo["blue"][1], histo["blue"][0], color = 'blue')
        plt.plot(histo["grey"][1], histo["grey"][0], color = 'black')
        plt.title('RGB histograms')
    
        if idx != None :
            plt.suptitle(str(idx)+' - '+label+' - Brightness = '+str(lum), fontsize = 'x-large')
        
        plt.tight_layout()
        plt.show()
        
    else:
        fig = plt.figure(figsize = (5,5))
        plt.imshow(img)
        
        if idx != None :
            plt.title(str(idx)+' - '+label+' - Brightness = '+str(lum), fontsize = 'x-large')
            plt.tight_layout()
            
        plt.show()
        
    return


def plot_random_class(df, label = None):
    """
    Affiche une cellule aléatoire du label renseigné
    """
    
    if label != None :
        filename = np.random.choice(df[df["label"] == label]["img_path"])
        idx = df[df["img_path"] == filename].index[0]
        print(idx)
    else :
        idx = np.random.choice(df.index)
        print(idx)
        filename = df.loc[idx,"img_path"]
        label = df.loc[idx,"label"]
        
    plot_luminance_cell(filename, label, idx)
    return

def plot_best_outliers(df, histos = True):
    """
    Affiche les images de luminosité moyenne minimale et maximale
    """
    idx_list = [df["mean_brightness"].argmax(),df["mean_brightness"].argmin()]
    
    for idx in idx_list:
        filename = df.loc[idx,"img_path"]
        label = df.loc[idx,"label"]
        plot_luminance_cell(filename, label, idx, histos)
    return

def plot_random_outliers(df, label = None):
    """
    Affiche des outliers au hasard (label = None), ou pour le type de cellule renseigné
    """
    if label != None:
        df_temp = df[df["label"] == label]
        df_temp = df_temp[np.abs(stats.zscore(df_temp["mean_brightness"])) > 2]
        
        idxs = list(np.random.choice(df_temp.index, replace = False, size = 4))
        filenames = list(df_temp.loc[idxs, "img_path"])
        labels = list(df_temp.loc[idxs, "label"])
        
        for filename, idx, label in zip(filenames, idxs, labels):
            plot_luminance_cell(filename, label, idx, histos = False)
            
    else:
        df_temp = df[np.abs(stats.zscore(df["mean_brightness"])) > 3]
        idxs = list(np.random.choice(df_temp.index, size = 4, replace = False))
        filenames = list(df_temp.loc[idxs,"img_path"])
        labels = list(df_temp.loc[idxs,"label"])

        for filename, idx in zip(filenames, idxs):
            plot_luminance_cell(filename, idx, histos = True)
    return


### Mean cells
def plot_mean_color_cell(df, labels):
    """
    Calcule et représente les cellules moyennes de chaque type.
    Retourne dans un DataFrame les données de chaque image moyenne.
    Gourmand en mémoire vive. Utiliser plot_mean_color_cell_ram si problème de RAM.
    
    df : pandas.DataFrame contenant les chemins d'accès et les labels
    labels : [] liste de labels
    """
    
    df_mean = pd.DataFrame(columns = ["r","g","b","grey","std_r","std_g","std_b","std_grey","std_rgb"])
    
    for label in labels:
        df_temp = df[df["label"] == label]['img_path']
        N = len(df_temp)
        print('Number of cells in %s : %i'%(label, N))

        images = np.array([leuko.load_image(filename) for filename in df_temp])
 
        # Calcul des images moyennes rouge, verte, bleue, rgb, gris
        mean_red = np.mean(images[:,:,:,0], axis = 0)
        mean_green = np.mean(images[:,:,:,1], axis = 0)
        mean_blue = np.mean(images[:,:,:,2], axis = 0)
        mean_rgb = np.mean(images, axis = 0)
        mean_grey = np.mean(images, axis = (0,3))
    
        # Calcul de l'écart-type
        std_red = np.std(images[:,:,:,0], axis = 0)
        std_green = np.std(images[:,:,:,1], axis = 0)
        std_blue = np.std(images[:,:,:,2], axis = 0)
        std_rgb = np.std(images, axis = 0)
        std_grey = np.std(images.mean(axis = 3), axis = 0)
    
        # Histogrammes de luminosité
        h_red = exposure.histogram(mean_red)
        h_green = exposure.histogram(mean_green)
        h_blue = exposure.histogram(mean_blue)
        h_grey = exposure.histogram(mean_grey)

        # Plot images
        plt.figure(figsize = (20,12))
    
        plt.subplot(3,5,1)
        plt.title("Mean "+label+" : Rouge")
        plt.imshow(mean_red, cmap = "Reds")
    
        plt.subplot(3,5,2)
        plt.title("Mean "+label+" : Vert")
        plt.imshow(mean_green, cmap = "Greens")
    
        plt.subplot(3,5,3)
        plt.title("Mean "+label+" : Bleu")
        plt.imshow(mean_blue, cmap = "Blues")
    
        plt.subplot(3,5,4)
        plt.title("Mean "+label+" : RGB")
        plt.imshow(mean_rgb)
    
        plt.subplot(3,5,5)
        plt.title("Mean "+label+" : Grey")
        plt.imshow(mean_red, cmap = "gray")
    
        # Plot STDS
        plt.subplot(3,5,6)
        plt.imshow(std_red)
        plt.title("Standard Deviation "+label+" : Red")
    
        plt.subplot(3,5,7)
        plt.imshow(std_green)
        plt.title("Standard Deviation "+label+" : Green")
    
        plt.subplot(3,5,8)
        plt.imshow(std_blue)
        plt.title("Standard Deviation "+label+" : Blue")
    
        plt.subplot(3,5,9)
        plt.imshow(std_rgb)
        plt.title("Standard Deviation "+label+" : RGB")
    
        plt.subplot(3,5,10)
        plt.imshow(std_grey)
        plt.title("Standard Deviation "+label+" : Grey")
    
        # Plot histogrammes
        plt.subplot(3,5,11)
        plt.plot(h_red[1], h_red[0], color = 'red')
        plt.title('Red-level histogram')
    
        plt.subplot(3,5,12)
        plt.plot(h_green[1], h_green[0], color = 'green')
        plt.title('Green-level histogram')
    
        plt.subplot(3,5,13)
        plt.plot(h_blue[1], h_blue[0], color = "blue")
        plt.title('Blue-level histogram')
    
        plt.subplot(3,5,14)
        plt.plot(h_red[1], h_red[0], color = "red")
        plt.plot(h_green[1], h_green[0], color = 'green')
        plt.plot(h_blue[1], h_blue[0], color = 'blue')
        plt.title('RGB histograms')
    
        plt.subplot(3,5,15)
        plt.plot(h_grey[1], h_grey[0], color = 'black')
        plt.title('Grey-level histogram')
        
        # Stockage de l'image moyenne dans le DataFrame df_mean
        df_mean.loc[label,"r"] = mean_red
        df_mean.loc[label,"g"] = mean_green
        df_mean.loc[label,"b"] = mean_blue
        df_mean.loc[label,"grey"] = mean_grey
        df_mean.loc[label,"std_r"] = std_red
        df_mean.loc[label,"std_g"] = std_green
        df_mean.loc[label,"std_b"] = std_blue
        df_mean.loc[label,"std_rgb"] = std_rgb   
        df_mean.loc[label,"std_grey"] = std_grey
    
        plt.show()
    return df_mean

def plot_mean_color_cell_ram(df, labels):
    """
    Calcule et représente les cellules moyennes de chaque type.
    Retourne dans un DataFrame les données de chaque image moyenne.
    df : pandas.DataFrame contenant les chemins d'accès et les labels
    labels : [] liste de labels
    """
    
    df_mean = pd.DataFrame(columns = ["r","g","b","grey"])
    
    for label in labels:
        df_temp = df[df["label"] == label]['img_path']
        N = len(df_temp)
        print('Number of cells in %s : %i'%(label, N))
              
        # Calcul des moyennes rouge, verte, bleue, rgb
        mean_red = np.zeros((363,360))
        mean_green = np.zeros((363,360))
        mean_blue = np.zeros((363,360))
        mean_rgb = np.zeros((363,360,3))
        mean_grey = np.zeros((363,360))
    
        for file in df_temp:
            image = leuko.load_image(file, as_grey = False, rescale = None, float32 = True)
            mean_red += image[:,:,0]
            mean_green += image[:,:,1]
            mean_blue += image[:,:,2]
            mean_grey += np.mean(image, axis = 2)
    
        mean_red = mean_red/N
        mean_green = mean_green/N
        mean_blue = mean_blue/N
        mean_grey = mean_grey/N
 
        df_mean.loc[label,"r"] = mean_red
        df_mean.loc[label,"g"] = mean_green
        df_mean.loc[label,"b"] = mean_blue
        df_mean.loc[label,"grey"] = mean_grey
    
    
        # On reconstitue l'image RGB
        mean_rgb[:,:,0] = mean_red
        mean_rgb[:,:,1] = mean_green
        mean_rgb[:,:,2] = mean_blue
    
        h_red = exposure.histogram(mean_red)
        h_green = exposure.histogram(mean_green)
        h_blue = exposure.histogram(mean_blue)
        h_grey = exposure.histogram(mean_grey)
    
        # Plot images
        plt.figure(figsize = (20,8))
    
        plt.subplot(2,5,1)
        plt.title("Mean "+label+" : Rouge")
        plt.imshow(mean_red, cmap = "Reds")
    
        plt.subplot(2,5,2)
        plt.title("Mean "+label+" : Vert")
        plt.imshow(mean_green, cmap = "Greens")
    
        plt.subplot(2,5,3)
        plt.title("Mean "+label+" : Bleu")
        plt.imshow(mean_blue, cmap = "Blues")
    
        plt.subplot(2,5,4)
        plt.title("Mean "+label+" : RGB")
        plt.imshow(mean_rgb)
    
        plt.subplot(2,5,5)
        plt.title("Mean "+label+" : Grey")
        plt.imshow(mean_grey, cmap = "gray")
    
        # Plot histogrammes
        plt.subplot(2,5,6)
        plt.plot(h_red[1], h_red[0], color = 'red')
        plt.title('Red-level histogram')
    
        plt.subplot(2,5,7)
        plt.plot(h_green[1], h_green[0], color = 'green')
        plt.title('Green-level histogram')
    
        plt.subplot(2,5,8)
        plt.plot(h_blue[1], h_blue[0], color = "blue")
        plt.title('Blue-level histogram')
    
        plt.subplot(2,5,9)
        plt.plot(h_red[1], h_red[0], color = "red")
        plt.plot(h_green[1], h_green[0], color = 'green')
        plt.plot(h_blue[1], h_blue[0], color = 'blue')
        plt.title('RGB histograms')
    
        plt.subplot(2,5,10)
        plt.plot(h_grey[1], h_grey[0], color = 'black')
        plt.title('Grey-level histogram')
    
    plt.show()
    return df_mean


def radial_luminance_mean_profile(df_mean, radius = None):
    """
    Profil de luminance radiale.
    df_mean : dataframe généré par plot_mean_color_cell_ram ou plot_mean_color_cell, contenant les données des
    images moyennes sur chaque canal + gris. Le calcul est fait sur les images en niveaux de gris.
    """
    
    plt.figure(figsize = (10,8))

    for label in df_mean.index:

        # Passage en coordonnées polaires, par rapport au centre de l'image d'origine (donc de la cellule)
        angular_img = skimage.transform.warp_polar(df_mean.loc[label,"grey"], radius = radius)
        radial_profile = angular_img.mean(axis = 0)
    
        # Barres d'erreur calculées par rapport à la variable angulaire (pas sur la population de cellules)
        radial_profile_std = angular_img.std(axis = 0)

        plt.errorbar(x = np.arange(0, angular_img.shape[1], 1, dtype = 'int32'),
                     y = radial_profile, 
                     yerr = radial_profile_std,
                     label = label, errorevery = 20)
        
    plt.xlim([0, angular_img.shape[1]])
    plt.xlabel('Coordonnée radiale')
    plt.ylabel('Luminosité')
    plt.title(' Profil radial de luminosité moyenne')
    plt.legend()
    plt.show()
    return

def radial_luminance_profile(df, labels, radius = 180):

    plt.figure(figsize = (10,8))
    j = 0
    for label in labels:
        df_temp = df[df["label"] == label]["img_path"]
        
        radial_profiles = np.zeros(shape = (radius,len(df_temp)))
        i = 0
    
        
        for filename in df_temp:
            
            # Passage en coordonnées polaires
            img = leuko.load_image(filename, as_grey = True)
            angular_img = skimage.transform.warp_polar(img, radius = 180)
            
            radial_profiles[:,i] = angular_img.mean(axis = 0)
            i += 1
            
            # Barres d'erreur calculées par rapport à la variable angulaire (pas sur la population de cellules)
            radial_profile = radial_profiles.mean(axis = 1)
            radial_profile_std = radial_profiles.std(axis = 1)
            
        plt.errorbar(
            x = np.arange(0, radius, 1, dtype = 'int32'),
            y = radial_profile,
            yerr = radial_profile_std,
            label = label, errorevery = 12+j)
        
        j+=1
        
    plt.xlim([0, radius])
    plt.xlabel('Coordonnée radiale')
    plt.ylabel('Luminosité moyenne (sur les cellules)')
    plt.title(' Profil radial de luminosité moyenne')
    plt.legend()
    plt.show()
    
    return

#### Déconvolution

def plot_deconv_single(idx, method=color.bpx_from_rgb):
    
    im = leuko.load_image(df.img_path[idx])
    lab = df.loc[idx]['label']
    deconv = color.separate_stains(im, method)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
    ax[0, 0].imshow(deconv[..., 0], cmap='Purples')
    ax[0, 0].set_title(f'{idx} - {lab}', fontweight='bold')
    hist, hist_centers = exposure.histogram(deconv[..., 0])
    ax[0, 1].plot(hist_centers, hist, lw=2, c='mediumpurple')
    ax[0, 1].set_title('channel 1')
    ax[0, 1].set_xlim(-0.1, 0.1)

    ax[1, 0].imshow(deconv[..., 1], cmap='pink_r')
    hist, hist_centers = exposure.histogram(deconv[..., 1])
    ax[1, 1].plot(hist_centers, hist, lw=2, c='gold')
    ax[1, 1].set_title('channel 2')
    ax[1, 1].set_xlim(-0.1, 0.1)

    
    ax[2, 0].imshow(deconv[..., 2], cmap='RdPu_r')
    hist, hist_centers = exposure.histogram(deconv[..., 2])
    ax[2, 1].plot(hist_centers, hist, lw=2, c='deeppink')
    ax[2, 1].set_title('channel 3')
    ax[2, 1].set_xlim(-0.1, 0.1)
    
    plt.show()
    return deconv, fig


def deconvolution_hist(filename, method=color.bpx_from_rgb):
    
    im = leuko.load_image(filename)
    temp = pd.DataFrame(index=[0])

    deconv = color.separate_stains(im, method)

    temp[['mean_C1', 'mean_C2', 'mean_C3']] = [np.mean(deconv, axis=(0, 1))]
    temp[['std_C1', 'std_C2', 'std_C3']] = [np.std(deconv, axis=(0, 1))]
    temp['hist0'] = [exposure.histogram(deconv[..., 0])]
    temp['hist1'] = [exposure.histogram(deconv[..., 1])]
    temp['hist2'] = [exposure.histogram(deconv[..., 2])]
    return temp

def box_plot(col, df):
    plt.figure(figsize = (16,6))
    plt.subplot(1,2,1)
    sns.boxplot(x = "label", y = col, data = df)

    plt.subplot(1,2,2)
    sns.kdeplot(x = col, hue = "label", data = df)
    
    plt.show()
    
def plot_histogram(df, title):
    hist0 = np.array([*df['hist0']]).mean(axis=0)
    hist1 = np.array([*df['hist1']]).mean(axis=0)
    hist2 = np.array([*df['hist2']]).mean(axis=0)

    plt.plot(hist0[1], hist0[0], c='mediumpurple', label='C1 (nucleus)')
    plt.plot(hist1[1], hist1[0], c='gold', label='C2 (rbc)')
    plt.plot(hist2[1], hist2[0], c='deeppink', label='C3 (bkg)')

    plt.xlim(-0.1, 0.1)
    plt.title(f'histogram for {title}')
    plt.legend()
    
def plot_histogram_per_cell(df, label):
    
    temp_df =  df[df.label == label]
    plot_histogram(temp_df, label)
    

def plot_deconv(idx, method):
    
    im = images[idx]
    deconv = color.separate_stains(im, method)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
    ax[0, 0].imshow(deconv[..., 0], cmap='Purples')
    hist, hist_centers = exposure.histogram(deconv[..., 0])
    ax[0, 1].plot(hist_centers, hist, lw=2)
    
    ax[1, 0].imshow(deconv[..., 1], cmap='pink_r')
    hist, hist_centers = exposure.histogram(deconv[..., 1])
    ax[1, 1].plot(hist_centers, hist, lw=2)
    
    ax[2, 0].imshow(deconv[..., 2], cmap='RdPu_r')
    hist, hist_centers = exposure.histogram(deconv[..., 2])
    ax[2, 1].plot(hist_centers, hist, lw=2)
    
    plt.show()

    return deconv, fig