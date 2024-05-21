##import the folowing libraries 
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage.color
import skimage.io 
from math import *
import skimage as sk
from skimage.measure import label,regionprops
from scipy.stats import gaussian_kde


#indicate path of .png files
path = r''
image_path_list = os.listdir(path) 

##looping images in defined folder
i = 0
for i in range (0,len(image_path_list)):
    image_path = image_path_list[i]
    image = skimage.io.imread(str(path)+"/" +str(image_path))
    #get mask of each section
    circular = create_circular_mask(image)
    #calculate histogram for a* channels on each section
    colour = [get_color_hist(mask) for mask in circular]
    print(colour)

    res = [*colour[0],
           *colour[1],
           *colour[2]]
    
    print(len(res))
    with open(path+"pattern_density.txt", "a") as temp_file_result:
        temp_file_result.write(str(image_path)+","+str(res)+"\n")
    i=+1


def create_circular_mask(image):

    binary_image = skimage.color.rgb2gray(image)
    binary_image[binary_image!=0]=255

    prop = sk.measure.regionprops(np.uint8(binary_image))
    centroid = prop[0].centroid
    y0, x0 = centroid[1], centroid[0]

    minor=prop[0].minor_axis_length/2
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(x0), int(y0)), int(minor), 255, -1)

    inner_mask = np.zeros(image.shape[:2], dtype="uint8")  
    cv2.circle(inner_mask, (int(x0), int(y0)), int(30/100*minor), 255, -1)
    
    mask_55 = np.zeros(image.shape[:2], dtype="uint8")  
    cv2.circle(mask_55, (int(x0), int(y0)), int(55/100*minor), 255, -1)
    
    mask_75 = np.zeros(image.shape[:2], dtype="uint8")  
    cv2.circle(mask_75, (int(x0), int(y0)), int(75/100*minor), 255, -1)
    
    part_inner = mask_55 - inner_mask
    section_inner = cv2.bitwise_and(image, image, mask=part_inner)

    part_medium = mask_75 - mask_55
    section_medium = cv2.bitwise_and(image, image, mask=part_medium)

    part_out = mask - mask_75
    section_out = cv2.bitwise_and(image, image, mask=part_out)


    return [section_inner,section_medium, section_out]

def show_lab(image):
    image_lab = skimage.color.rgb2lab(image)
    # Lab Color Space - https://en.wikipedia.org/wiki/CIELAB_color_space
    L = image_lab[:, :, 0]
    a = image_lab[:, :, 1]
    b = image_lab[:, :, 2]
    fig, ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(L)
    ax[1].imshow(a)
    ax[2].imshow(b)
    plt.show()

def show_hsv(image):
    image_lab = skimage.color.rgb2hsv(image)
    # Lab Color Space - https://en.wikipedia.org/wiki/CIELAB_color_space
    h = image_lab[:, :, 0]
    s = image_lab[:, :, 1]
    v = image_lab[:, :, 2]
    fig, ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(h)
    ax[1].imshow(s)
    ax[2].imshow(v)
    plt.show()


def get_color_hist(image):
    #color conversion in the following color space : RGB, L*a*bù* and HSV.
    image_lab = skimage.color.rgb2lab(image)

    a = image_lab[:, :, 1]
    a_no_bg = a != 0

    b = image_lab[:, :, 2]
    b_no_bg = b != 0

    plt.hist(a[a_no_bg], bins =6, range= (0,60))

    frequency, bins = np.histogram(a[a_no_bg], bins =6)
    bins=np.delete(bins,0)

    return(frequency)




def gaussian_density(image) : 
    #color conversion in the following color space : RGB, L*a*bù* and HSV.
    image_lab = skimage.color.rgb2lab(image)

    L = image_lab[:, :, 0]
    L_no_bg = L > 0
   
    a = image_lab[:, :, 1]
    a_no_bg = a != 0
   
    b = image_lab[:, :, 2]
    b_no_bg = b != 0
   
    colour_range = [i for i in range(1, 101)]
    
    #automatic partition according to min/max of L* values
    #eval_points_L = np.linspace(np.min(L[L_no_bg]), np.max(L[L_no_bg]))

    kde_sp_L = gaussian_kde(L[L_no_bg], bw_method=0.25)
    y_sp_L = kde_sp_L.pdf(colour_range)  

    #automatic partition according to min/max of a* values
    #eval_points_a = np.linspace(np.min(a[a_no_bg]), np.max(a[a_no_bg]))


    kde_sp_a = gaussian_kde(a[a_no_bg], bw_method=0.25)
    y_sp_a = kde_sp_a.pdf(colour_range)  


    #automatic partition according to min/max of b* values
    #eval_points_b = np.linspace(np.min(b[b_no_bg]), np.max(b[b_no_bg]))

    kde_sp_b = gaussian_kde(b[b_no_bg], bw_method=0.25)
    y_sp_b = kde_sp_b.pdf(colour_range)  


    return(y_sp_L,y_sp_a,y_sp_b)

