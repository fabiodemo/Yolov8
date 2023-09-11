#!/usr/bin/env python
# coding: utf-8

# # **Testing on a Demo Video**

# In[ ]:


get_ipython().system('gdown "https://drive.google.com/uc?id=1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ&confirm=t" -o content/demo.mp4')


# In[ ]:


get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system("yolo task=detect mode=predict model={HOME}/runs/detect/train8/weights/best.pt conf=0.25 source='content/demo.mp4'")


# ## **Display the Demo Video**

# In[ ]:


from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = 'runs/detect/predict2/demo.avi'

# Compressed video path
compressed_path = "content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

