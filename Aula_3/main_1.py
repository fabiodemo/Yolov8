#!/usr/bin/env python
# coding: utf-8

# # **Importing the Required Libraries**

# In[1]:


import os
import glob


# In[2]:


# If we want to display a single image, then "Image" Library is fine, but if we want to display multiple images by just running a single cell
# then we need to import display

from IPython.display import Image, display


# In[ ]:


# To clear output in the notebook use clear_output function
#display.clear_output()


# # **In the First Step, We need to check whether we have access to the GPU or not**

# In[3]:


get_ipython().system('nvidia-smi')


# In[7]:


HOME = os.getcwd()


# In[8]:


print(HOME)


# ## **Installing Ultralytics using Pip Install**

# **YOLOv8 can be installed in two ways - from the source and via pip. This is because it is the first iteration of YOLO to have an official package.**

# In[ ]:


# Git clone method (for development)

# %cd {HOME}
# !git clone github.com/ultralytics/ultralytics
# %cd {HOME}/ultralytics
# !pip install -qe ultralytics

# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()


# In[6]:


#!pip install ultralytics==8.0.0


# ## Checking whether YOLOv8 is Installed and its working Fine

# In[5]:


import ultralytics


# In[6]:


ultralytics.checks()


# # **Importing the PPE Detection Dataset from Roboflow**

# In[9]:


get_ipython().system('mkdir {HOME}/datasets')


# In[10]:


get_ipython().system('pwd')


# In[11]:


get_ipython().run_line_magic('cd', '{HOME}/datasets')


# In[12]:


get_ipython().system('pwd')


# In[14]:


# https://universe.roboflow.com/object-detection/eep_detection-u9bbd
# !pip install roboflow

#!pip install roboflow
#from roboflow import Roboflow
#rf = Roboflow(api_key="4hIhYKGrnWHaWXRqZsZg")
#project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
#dataset = project.version(1).download("yolov5")


# In[15]:


from roboflow import Roboflow
rf = Roboflow(api_key="zI9OTjbwPrkuQDUQXD8i")
project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
dataset = project.version(1).download("yolov8")


# # **Train the YOLOv8 Model on the Custom Dataset**

# In[16]:


get_ipython().run_line_magic('cd', '{HOME}')


# In[17]:


get_ipython().run_line_magic('cd', '{dataset.location}')


# If you want to train, validate or run inference on models and don't need to make any modifications to the code, using YOLO command line interface is the easiest way to get started. Read more about CLI in [Ultralytics YOLO Docs](https://v8docs.ultralytics.com/cli/).
# 
# ```
# yolo task=detect    mode=train    model=yolov8n.yaml      args...
#           classify       predict        yolov8n-cls.yaml  args...
#           segment        val            yolov8n-seg.yaml  args...
#                          export         yolov8n.pt        format=onnx  args...
# ```

# In[25]:


get_ipython().run_line_magic('cd', '{HOME}')

get_ipython().system('yolo task=detect mode=train model=yolov8m.pt data={HOME}/datasets/EEP_Detection-1/data.yaml epochs=90 imgsz=640')


# In[ ]:



#!zip -r /content/rede_treinada.zip /content/runs/


# In[ ]:


#!rm -r /content/content/
#!rm -r /content/rede_treinada.zip


# In[7]:


# !unrar x /content/runs.part1.rar  /content/


# In[9]:


get_ipython().system('ls {HOME}/runs/detect/train8')


# # Validate custom model

# In[11]:


#Here, we are taking the model best weights and using them to validate the model. Similarly as before we are using
#CLI to do that, The only difference is our mode = val instead of train
#Validation Script is using test dataset that was not used beforew

get_ipython().run_line_magic('cd', '{HOME}')

get_ipython().system('yolo task=detect mode=val model={HOME}/runs/detect/train8/weights/best.pt data={HOME}/datasets/EEP_Detection-1/data.yaml')


# # **Displaying the Confusion Matrix**

# In[13]:


#Confusion matrix is the chart that shows how our model handles different classes
#92% of the time the model detected correctly that the person is wearing jacket, while 1% of the time we get the Bounding Box but
#the jacket is incorrectly classified as Eye wear, while 7% of the time when person is wearing the Jacket the model is unable to detect it
get_ipython().run_line_magic('cd', '{HOME}')
Image(filename=f'{HOME}/runs/detect/val/confusion_matrix_normalized.png', width=900)


# # **Training and Validation Loss**

# In[14]:


# Here is the graph of the training and validation loss
#box loss and class loss is important
# The behavior of the model is convincing the model is coverging, Training more will give better results
get_ipython().run_line_magic('cd', '{HOME}')

Image(filename=f'{HOME}/runs/detect/train8/results.png', width=600)


# In[16]:


#Model Prediction on validation batch. These image are not used strictly for training so it is always better to take a
#look and see how model is behaving
get_ipython().run_line_magic('cd', '{HOME}')
Image(filename=f'{HOME}/runs/detect/val/val_batch0_pred.jpg', width=600)


# # **Inference with Custom Model**

# In[19]:


#Inference means a prediction that we can run on an image to detect the label,
# whether classification or of a bounding box or a segmentation
# Testing the Model on Test Dataset images
get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system("yolo task=detect mode=predict model={HOME}/runs/detect/train8/weights/best.pt conf=0.25 source='datasets/EEP_Detection-1/test/images'")


# In[20]:


import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:5]:
      display(Image(filename=image_path, width=600))
      print("\n")

