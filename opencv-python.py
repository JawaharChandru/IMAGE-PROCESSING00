#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt


# In[2]:


def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# In[4]:


image = cv2.imread(r"C:\Users\admin\Downloads\forest.jpg")

imshow("Castara, Tobago", image)


# In[10]:


image.shape[:2] # (1200, 1920) (height, width)
def imshow(title = "", image = None, size = 10):
 
    # The line below is changed from w, h to h, w
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
 
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

imshow("Castara, Tobago", image)


# In[11]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imshow("Converted to Grayscale", gray_image)


# In[25]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define our imshow function 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# In[23]:


image = cv2.imread(r"C:\Users\admin\Downloads\forest.jpg")

# Use cv2.split to get each color space separately
B, G, R = cv2.split(image)
print(B.shape)
print(G.shape)
print(R.shape)


# In[24]:


imshow("Blue Channel Only", B)


# In[26]:


import numpy as np

# Let's create a matrix of zeros 
# with dimensions of the image h x w  
zeros = np.zeros(image.shape[:2], dtype = "uint8")

imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))


# In[27]:


image = cv2.imread(r"C:\Users\admin\Downloads\forest.jpg")

# OpenCV's 'split' function splites the image into each color index
B, G, R = cv2.split(image)

# Let's re-make the original image, 
merged = cv2.merge([B, G, R]) 
imshow("Merged", merged) 


# In[28]:


merged = cv2.merge([B+100, G, R])
imshow("Blue Boost", merged)


# In[20]:


image = cv2.imread(r"C:\Users\admin\Downloads\forest.jpg")

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)


# In[21]:


plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
plt.show()


# In[22]:


imshow("Hue", hsv_image[:, :, 0])
imshow("Saturation", hsv_image[:, :, 1])
imshow("Value", hsv_image[:, :, 2])


# In[31]:


image = np.zeros((512,512,3), np.uint8)
image_gray = np.zeros((512,512), np.uint8)
imshow("Black Canvas - RGB Color", image)
imshow("Black Canvas - Grayscale", image_gray)


# In[32]:


cv2.line(image, (0,0), (511,511), (255,127,0), 5)

imshow("Black Canvas With Diagonal Line", image)


# In[33]:


image = np.zeros((512,512,3), np.uint8)
cv2.rectangle(image, (100,100), (300,250), (127,50,127), 10)
imshow("Black Canvas With Pink Rectangle", image)


# In[34]:


image = np.zeros((512,512,3), np.uint8)

cv2.circle(image, (350, 350), 100, (15,150,50), -1) 
imshow("Black Canvas With Green Circle", image)


# In[35]:


image = np.zeros((512,512,3), np.uint8)
pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(image, [pts], True, (0,0,255), 3)
imshow("Black Canvas with Red Polygon", image)


# In[37]:


image = np.zeros((1000,1000,3), np.uint8)
ourString =  'JAWAHAR JC'
cv2.putText(image, ourString, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40,200,0), 4)
imshow("Messing with some text", image)


# In[ ]:




