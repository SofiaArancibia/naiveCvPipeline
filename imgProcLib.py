#!/usr/bin/env python
# coding: utf-8

# ### **Decisions**:
#     - I did the unzipping manually
# 
# ### **Assumptions**:
#     - I don't have duplicated images
#     - The samples are always going to be squared (NxN)
#     - The statement: "Ensure each image is 480 x 270x 3(Resize if required)", is traslated as follows:
#                         * 480 -> widht
#                         * 270 -> height
#                         * 3 -> depth (channels)
# 
# ### **Improvements**:
#     - Process images (resizing and cropping) in grayscale to reduce space and compute complexity
#     - Create an external samples folder to store image samples. A good practice is to save raw, interim and processed data for experiments reproducibility purposes. [Already applied for testing purposes]

# In[ ]:


from pathlib import Path
import cv2
import numpy as np
from random import randint
from random import shuffle
import math


# In[ ]:


cv2.__version__


# In[ ]:


def read_image(path):    
    '''
    description:
        Reads an image from its path
    input:
        - path: string
    output:
        - img: cv2 image (array)
    '''    
    img = cv2.imread(path)    
    return img


# In[ ]:


def check_image_size(img, size):
    '''
    description:
        Returns True or False depending on the image having a specific size passed as argument
    input:
        - img: cv2 image
        - size: list [height, width, depth]
    output:
        - check: boolean
    '''
    #TODO: check if 'size' has 3 elements
    
    height, width, depth = img.shape    
    
    return (height, width, depth) == tuple(size)


# In[ ]:


def resize_image(img, size):
    '''
    description:
        Resizes an image 'img' to the 'size' size passed as argument
    input:
        - img: cv2 image
        - size: list/tuple [height, width, depth]
    output:
        - resized: cv2 image
    '''       
    
    resized = cv2.resize(img, (size[0], size[1]), interpolation = cv2.INTER_AREA)    
    
    print("Original Size: {}".format(img.shape))
    print("Resized Size: {}".format(resized.shape))
    
    return resized


# In[ ]:


def crop_image(img, roi_size, shift, center=True):
    
    #TODO: allow to center=False
    #TODO: allow specify a point as 'shift' value
    
    '''
    description: Crops an image to size 'roi_size'. 
    The center of the image is the default shift, but if center = False, a shift value (a new center) must be provided.
    input:
        - img: cv2 image
        - roi_size: list [roi_height, roi_width]
        - shift: list [y_shift, x_shift]
        - center: boolean 
    output:
        - crop_img: cv image
    '''
    
    height, width, _ = img.shape  
    img_center = [int(height/2), int(width/2)]
    if center:
        crop_img = img[img_center[0]-int(roi_size[0]/2):img_center[0]+int(roi_size[0]/2), img_center[1]-int(roi_size[1]/2):img_center[1]+int(roi_size[1]/2)]
    else:
        if shift:
            pass
        else:
            raise ValueError('shift value must be setted when center=False')
    
    print("Original Size: {}".format(img.shape))
    print("Cropped Size: {}".format(crop_img.shape))
    
    return crop_img


# In[ ]:


def get_random_corner(sample_shape, img_shape):
    
    '''
    description: Returns a point in the image. As the point is going to be used as sample top-left corner, the function
    also checks if the final sample is crossing the boundaries. If so, re-calculates the point.
    input:
        - sample_shape: list
        - img_shape: cv2 image
    output:
        - corner: list
    '''
    
    ylimit = img_shape[0]
    xlimit = img_shape[1]
    
    corner = [randint(0, ylimit), randint(0, xlimit)]
    
    ycheck = (corner[0] + sample_shape[0]) <= ylimit
    xcheck = (corner[1] + sample_shape[1]) <= xlimit
    
    while ((not ycheck) or (not xcheck)):
        corner = get_random_corner(sample_shape, img_shape)
        ycheck = (corner[0] + sample_shape[0]) <= ylimit
        xcheck = (corner[1] + sample_shape[1]) <= xlimit
    
    return corner

def check_corners(corner, corner_lists, sample_size, img_shape):
    
    '''
    description: Checks potential samples overlapping by measuring sample center distance.
    input:
        - corner: list
        - corner_lists: list
        - sample_size: list
        - img_shape: list
    output:
        - overlap: boolean
    '''
    
    # Checking overlaping by centers distance
    # For a non-overlaping case, the distance between the centers should be > sample size 
    # Considering the samples are all NxN
    
    # Current corner center 
    corner_center_y_coord = corner[0] + int(sample_size[0]/2)
    corner_center_x_coord = corner[1] + int(sample_size[1]/2)
        
    for c in corner_lists:
        overlap = False
        c_center_y_coord = c[0] + int(sample_size[0]/2)
        c_center_x_coord = c[1] + int(sample_size[1]/2)
        
        # Euclidean distance
        dist = math.sqrt(((c_center_y_coord - corner_center_y_coord)**2) + ((c_center_x_coord - corner_center_x_coord)**2))
        
        if (dist <= sample_size[0]): # This is why I assume that the samples are always NxN. 
            overlap = True           # If not, the check should consider both distances (to y_limit and to x_limit)           
            return overlap
    return overlap         
        

def extract_samples(img, sample_size, overlap=False, num_samples=3):
    '''
    description: Extract 'num_samples' samples of 'sample_size' size from 'img'. 
    input:
        - img: cv2 image
        - num_samples: int
        - sample_size: list [sample_height, sample_width]
        - overlap: boolean 
    output:
        - samples: cv images list
    '''
    
    corners = list()
    samples = list()
    img_size = img.shape

    corner = get_random_corner(sample_size, img_size) # Pick a corner
    corners.append(corner) # Store it
    while(len(corners)<num_samples): # Keep picking corners until 'num_samples' of corners are collected
        if not overlap:
            corner = get_random_corner(sample_size, img_size) # Pick a corner
            check = check_corners(corner, corners, sample_size, img_size) # Check potential overlapping
            while(check): # While the corners overlaps, keep trying with other
                corner = get_random_corner(sample_size, img_size)
                check = check_corners(corner, corners, sample_size, img_size)
            corners.append(corner)
        else: # This should be completed if we do want to allow samples overlaping (overlap=True)
            pass
    
    # Extract the samples
    for c in corners:
        sample = img[c[0]:c[0]+sample_size[0],c[1]:c[1]+sample_size[1]]
        samples.append(sample)      
    
    return samples    


# In[ ]:


def train_test_split(data, test = 0.3, allow_shuffle = True):
    '''
    description: Splits a dataset of samples into training and testing sets. 
    input:
        - data: dict
        - test: float
        - shuffle: boolean
    output:
        - train: list
        - test: list
    '''
    
    points = len(data)
    test_len = int(test*points)
    keys =  list(data.keys())
    train = list()
    test = list()
    counter = 0
    
    if allow_shuffle:
        shuffle(keys)
    
    for key in keys:
        if counter<=test_len:
            test.extend(data[key])
            counter += 1
        else:
            train.extend(data[key])
    
    return train, test    


# In[ ]:


def save_samples(folder, img_name, sample_list):
    '''
    description: Save samples to a specific folder. 
    input:
        - folder: string
        - img_name: string
        - sample_list: list
    output:
        None
    '''
    counter = 0
    for sample in sample_list:
        path = folder + "/" + img_name + "_sample{}.jpeg".format(counter)
        print(path)
        cv2.imwrite(path, sample)
        counter += 1

