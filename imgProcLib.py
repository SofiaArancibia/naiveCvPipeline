#!/usr/bin/env python
# coding: utf-8

# ### **Decisions**:
#     - I did the unzipping manually
# 
# ### **Assumptions**:
#     - I don't have duplicated images
# 
# ### **Improvements**:
#     - Process images (resizing and cropping) in grayscale to reduce space and compute complexity
#     - Create an external samples folder to store image samples. A good practice is to save raw, interim and processed data for experiments reproducibility purposes.

# In[1]:


from pathlib import Path
import cv2
import numpy as np
from random import randint
from random import shuffle


# In[2]:


def read_image(path):
    img = cv2.imread(file)
    return img


# In[3]:


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
    (height, width, depth) = img.shape    
    return (height, width, depth) == tuple(size)


# In[4]:


def resize_image(img, size):
    #TODO: check if 'size' has 3 elements
    
    if (isinstance(size, list)):
        size = tuple(size)
    
    print(size)
    
    resized = cv2.resize(img, size)
    print("Original Size: {}".format(img.shape))
    print("Resized Size: {}".format(resized.shape))
    return resized


# In[5]:


def crop_image(img, roi_size, shift, center=True):
    #TODO: check if 'roi_size' has 3 elements
    #TODO: allow to center=False
    #TODO: allow specify a corner as 'shift' value
    #TODO: check if roi_size is under the image boundaries
    
    '''
    description: Crops an image to size 'roi_size'. 
    The center of the image is the default shift, but if center = False, a shift value must be provided in the form [center+y, center+x]
    input:
        - img: cv2 image
        - roi_size: list [roi_height, roi_width]
        - shift: list [y_shift, x_shift]
        - center: boolean 
    output:
        - crop_img: cv image
    '''
    height, width, _ = img.shape  
    img_center = [height/2, width/2]
    if center:
        crop_img = img[img_center[0]-roi_size[0]/2:img_center[0]+roi_size[0]/2, img_center[1]-roi_size[1]/2:img_center[1]+roi_size[1]/2]
    else:
        if shift:
            pass
        else:
            raise ValueError('shift value must be setted when center=False')
    
    print("Original Size: {}".format(img.shape))
    print("Cropped Size: {}".format(crop_img.shape))
    return crop_img


# In[6]:




def get_random_corner(y_limit, x_limit):
   return [randint(0, y_limit), randint(0, x_limit)]

def check_corners(corner, corner_lists, sample_size):
   for c in corner_lists:
       ly = c[0] + sample_size[0]
       lx = c[1] + sample_size[1]
       overlap = False
       if ((corner[0] >= c[0] and corner[0] < ly) or ( corner[0] > c[0] and corner[0] <= ly)) and ((corner[1] >= c[1] and corner[1] < lx) or ( corner[1] > c[1] and corner[1] <= lx)):           
           overlap = True    
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
   for i in range(0,num_samples):
       corner = get_random_corner(sample_size[0], sample_size[1])
       corners.append(corner)
       if not overlap:
           corner = get_random_corner(sample_size[0], sample_size[1])
           check = check_corners(corner, corners, sample_size)
           while(check):
               corner = get_random_corner(sample_size[0], sample_size[1])
           corners.append(corner)
       else:
           pass
   
   for c in corners:
       sample = img[c[0]:c[0]+sample_size[0],c[1]:c[1]+sample_size[1]]
       samples.append(sample)      
   
   return samples    


# In[7]:


def train_test_split(data, test = 0.3, shuffle = True):
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
    
    if shuffle:
        shuffle(keys)
    
    for key in keys:
        if counter<=test_len:
            test.extend(data[key])
        train.extend(data[key])
    
    return train, test    


# # Pipeline usecase

# In[9]:


import glob
images_path = "ML Engineer task/ML Engineer task/sample_frames/sample_frames/*"
files = glob.glob(images_path)

img_size = [480, 270, 3]
roi_size = [270, 270, 3]
sample_size = [80, 80, 3]

samples = dict()
counter = 0
for file in files:
    print("Processing file: {}".format(file))
    img = cv2.imread(file) #TODO: check if 'file' is an image
    cv2.imshow("Read image", img)
    check = check_image_size(img, img_size) # Ensure each image is 480 x 270 x 3
    
    if not check: # Resize if required
        img = resize_image(img, img_size) 
    
    # Crop the image down to the central 270x270x3 region
    img = crop_image(img, roi_size, _, center=True)
    
    # Randomly extract 3, 80 x 80 x 3 samples that do not overlap
    samples['img{}_samples'.format(counter)] = extract_samples(img, sample_size, overlap=False, num_samples=3)
    counter += 1
    
# Allow for shuffling & separation into training & test sets. 
# The proportions of which should be able to be defined by the end user. 
# Samples from the same image should not appear in both training and test sets.

train, test  = train_test_split(samples, test = 0.3, shuffle = True)    


# In[ ]:




