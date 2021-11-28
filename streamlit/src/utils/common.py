from skimage import transform, img_as_float32
import skimage


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
        image = transform.resize(image, (363, 360))  # resize outliers

    else:
        image = skimage.io.imread(filename)
        image = transform.resize(image, (363, 360, 3))  # resize outliers

    if rescale:
        image = transform.rescale(
            image, rescale, anti_aliasing=True)  # reduce dim

    if float32:
        # Optional: set to your desired precision
        image = img_as_float32(image)

    return image
