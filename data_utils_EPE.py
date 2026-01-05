import numpy as np
from scipy.ndimage import zoom


def resize_image(ori_img, output_shape = [8,64,64]):
    
    output_img = np.zeros((len(ori_img),output_shape[0],output_shape[1],output_shape[2]))

    for i in range(len(ori_img)):
        img = np.array(ori_img[i])
        size = img.shape
        output_img[i] = zoom(img, (output_shape[0]/size[0], output_shape[1]/size[1], output_shape[2]/size[2]))

    return output_img

def resize_img(ori_img , output_shape):
    ori_img_size = ori_img.shape
    out_img = zoom(ori_img, (output_shape[0]/ori_img_size[0], output_shape[1]/ori_img_size[1], output_shape[2]/ori_img_size[2]))
    return out_img
