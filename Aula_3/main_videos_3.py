#!/usr/bin/env python
# coding: utf-8

# # **Testing the Demo Video 3**

# In[ ]:


get_ipython().system('gdown "https://drive.google.com/uc?id=1256pNK0nQnEDT6FRLQAraTRkOY7BSprq&confirm=t" -O content/demo3.mp4')


# ## **Testing on the Demo Video 3**

# In[ ]:


get_ipython().run_line_magic('cd', '{HOME}')
get_ipython().system("yolo task=detect mode=predict model={HOME}/runs/detect/train8/weights/best.pt conf=0.25 source='content/demo3.mp4'")


# ## **Display the Demo Video**

# In[ ]:


from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = 'runs/detect/predict4/demo3.avi'

# Compressed video path
compressed_path = "content/demo3_result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)


# In[ ]:


#saving in .rar  all generated files of the project
# !apt-get install rar
get_ipython().system('rar a "content/all_files_generated.rar" "content/"')


# In[ ]:


get_ipython().system('rm content/*.mp4')

