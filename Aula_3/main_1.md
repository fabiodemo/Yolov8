# **Importing the Required Libraries**


```python
import os
import glob
```


```python
# If we want to display a single image, then "Image" Library is fine, but if we want to display multiple images by just running a single cell
# then we need to import display

from IPython.display import Image, display
```


```python
# To clear output in the notebook use clear_output function
#display.clear_output()
```

# **In the First Step, We need to check whether we have access to the GPU or not**


```python
!nvidia-smi
```

    Tue Sep  5 20:40:14 2023       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce RTX 3060         On | 00000000:09:00.0 Off |                  N/A |
    |  0%   52C    P8               17W / 170W|    610MiB / 12288MiB |     28%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A      1168      G   /usr/lib/xorg/Xorg                           47MiB |
    |    0   N/A  N/A      1674      G   /usr/lib/xorg/Xorg                          210MiB |
    |    0   N/A  N/A      1811      G   /usr/bin/gnome-shell                         47MiB |
    |    0   N/A  N/A      4487      G   ...7330179,17326161306471323119,262144       64MiB |
    |    0   N/A  N/A      6609      G   ...sion,SpareRendererForSitePerProcess      141MiB |
    |    0   N/A  N/A     21623      G   ...--disable-features=BackForwardCache       73MiB |
    +---------------------------------------------------------------------------------------+
    


```python
HOME = os.getcwd()
```


```python
print(HOME)
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    

## **Installing Ultralytics using Pip Install**

**YOLOv8 can be installed in two ways - from the source and via pip. This is because it is the first iteration of YOLO to have an official package.**


```python
# Git clone method (for development)

# %cd {HOME}
# !git clone github.com/ultralytics/ultralytics
# %cd {HOME}/ultralytics
# !pip install -qe ultralytics

# from IPython import display
# display.clear_output()

# import ultralytics
# ultralytics.checks()
```


```python
#!pip install ultralytics==8.0.0
```

## Checking whether YOLOv8 is Installed and its working Fine


```python
import ultralytics
```


```python
ultralytics.checks()
```

    Ultralytics YOLOv8.0.170 🚀 Python-3.9.18 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060, 12042MiB)
    Setup complete ✅ (8 CPUs, 23.4 GB RAM, 3.3/29.8 GB disk)
    

# **Importing the PPE Detection Dataset from Roboflow**


```python
!mkdir {HOME}/datasets
```


```python
!pwd
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    


```python
%cd {HOME}/datasets

```

    /home/fabio/Documents/Github/Yolov8/Aula_3/datasets
    


```python
!pwd
```

    /home/fabio/Documents/Github/Yolov8/Aula_3/datasets
    


```python
# https://universe.roboflow.com/object-detection/eep_detection-u9bbd
# !pip install roboflow

#!pip install roboflow
#from roboflow import Roboflow
#rf = Roboflow(api_key="4hIhYKGrnWHaWXRqZsZg")
#project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
#dataset = project.version(1).download("yolov5")
```


```python
from roboflow import Roboflow
rf = Roboflow(api_key="zI9OTjbwPrkuQDUQXD8i")
project = rf.workspace("objet-detect-yolov5").project("eep_detection-u9bbd")
dataset = project.version(1).download("yolov8")
```

    loading Roboflow workspace...
    loading Roboflow project...
    Dependency ultralytics==8.0.134 is required but found version=8.0.170, to fix: `pip install ultralytics==8.0.134`
    Downloading Dataset Version Zip in EEP_Detection-1 to yolov8: 100% [113214287 / 113214287] bytes
    

    Extracting Dataset Version Zip to EEP_Detection-1 in yolov8:: 100%|██████████| 6482/6482 [00:00<00:00, 7195.70it/s]
    

# **Train the YOLOv8 Model on the Custom Dataset**


```python
%cd {HOME}
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    


```python
%cd {dataset.location}
```

    /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1
    

If you want to train, validate or run inference on models and don't need to make any modifications to the code, using YOLO command line interface is the easiest way to get started. Read more about CLI in [Ultralytics YOLO Docs](https://v8docs.ultralytics.com/cli/).

```
yolo task=detect    mode=train    model=yolov8n.yaml      args...
          classify       predict        yolov8n-cls.yaml  args...
          segment        val            yolov8n-seg.yaml  args...
                         export         yolov8n.pt        format=onnx  args...
```


```python
%cd {HOME}

!yolo task=detect mode=train model=yolov8m.pt data={HOME}/datasets/EEP_Detection-1/data.yaml epochs=90 imgsz=640
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    New https://pypi.org/project/ultralytics/8.0.171 available 😃 Update with 'pip install -U ultralytics'
    Ultralytics YOLOv8.0.170 🚀 Python-3.9.18 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060, 12037MiB)
    [34m[1mengine/trainer: [0mtask=detect, mode=train, model=yolov8m.pt, data=/home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/data.yaml, epochs=90, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, stream_buffer=False, line_width=None, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train8
    Overriding model.yaml nc=80 with nc=7
    
                       from  n    params  module                                       arguments                     
      0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
      1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
      2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
      3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
      4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
      5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
      6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
      7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
      8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
      9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
     16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
     19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
     22        [15, 18, 21]  1   3779749  ultralytics.nn.modules.head.Detect           [7, [192, 384, 576]]          
    Model summary: 295 layers, 25860373 parameters, 25860357 gradients
    
    Transferred 469/475 items from pretrained weights
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to 'yolov8n.pt'...
    100%|██████████████████████████████████████| 6.23M/6.23M [00:00<00:00, 10.9MB/s]
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1mtrain: [0mScanning /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detectio[0m
    [34m[1mtrain: [0mNew cache created: /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/train/labels.cache
    [34m[1mval: [0mScanning /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-[0m
    [34m[1mval: [0mNew cache created: /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/valid/labels.cache
    Plotting labels to runs/detect/train8/labels.jpg... 
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1498: FutureWarning: is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
      if pd.api.types.is_categorical_dtype(vector):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /home/fabio/mambaforge/envs/yolo/lib/python3.9/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    [34m[1moptimizer:[0m AdamW(lr=0.000909, momentum=0.9) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)
    Image sizes 640 train, 640 val
    Using 8 dataloader workers
    Logging results to [1mruns/detect/train8[0m
    Starting training for 90 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           1/90      6.43G      1.347      1.789      1.372         45        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.692      0.574      0.616      0.384
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           2/90      6.65G      1.305        1.2      1.339         65        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.753      0.497      0.565      0.329
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           3/90      6.66G      1.318       1.15      1.354         42        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.787      0.553      0.593      0.364
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           4/90      6.64G      1.272      1.085      1.323         53        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.667      0.681      0.646      0.398
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           5/90      6.65G      1.242     0.9956      1.297         46        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.869      0.648      0.727      0.452
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           6/90      6.64G       1.23     0.9694      1.287         64        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.776      0.649      0.712      0.438
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           7/90      6.64G      1.192     0.9169       1.27         67        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074        0.9      0.673      0.743      0.473
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           8/90      6.62G      1.179     0.8719      1.259         72        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.816      0.711      0.769      0.492
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
           9/90      6.64G      1.151     0.8618      1.251         71        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.859      0.727      0.808      0.517
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          10/90      6.65G      1.142     0.8355      1.241         57        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.784      0.788      0.815      0.517
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          11/90      6.65G      1.117     0.7972      1.219         62        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.843      0.745      0.797      0.525
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          12/90      6.65G      1.109      0.766      1.218         73        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.818      0.797      0.826      0.537
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          13/90      6.65G      1.104     0.7685       1.22         67        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.829      0.763      0.802      0.519
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          14/90      6.65G      1.064     0.7389      1.199         44        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.862      0.749      0.821      0.549
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          15/90      6.66G      1.082     0.7327      1.199         54        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.837      0.768      0.825      0.542
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          16/90      6.61G      1.055     0.7056       1.19         35        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.872      0.762      0.831      0.547
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          17/90      6.65G      1.053     0.6904      1.186         69        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.835      0.796      0.832      0.554
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          18/90      6.64G       1.03     0.6788      1.179         50        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.855      0.804      0.859      0.581
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          19/90      6.66G      1.024     0.6613      1.165         51        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.868      0.805      0.846      0.564
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          20/90      6.64G      1.018     0.6485      1.162         43        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.875      0.807      0.861      0.579
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          21/90      6.65G      1.006     0.6455      1.151         64        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.911      0.793      0.863      0.581
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          22/90      6.64G      1.002     0.6402      1.158         64        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.913      0.801      0.869      0.586
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          23/90      6.65G     0.9946     0.6093      1.144         51        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.898      0.825      0.884      0.598
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          24/90      6.64G     0.9854     0.6238      1.143         47        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.876      0.821      0.862      0.589
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          25/90      6.65G     0.9942     0.6202      1.148         48        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.884      0.822      0.865      0.583
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          26/90      6.61G     0.9797     0.6169      1.144         86        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.921      0.812       0.88      0.595
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          27/90      6.65G     0.9546     0.6063      1.136         55        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.915      0.819      0.883      0.604
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          28/90      6.62G     0.9465     0.5865      1.124         50        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.861      0.846      0.878      0.608
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          29/90      6.64G     0.9417     0.5716      1.112         67        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.878       0.82      0.869      0.598
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          30/90      6.64G     0.9485     0.5752      1.126         81        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.879      0.829      0.883      0.604
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          31/90      6.65G     0.9403     0.5694      1.119         24        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.937      0.816      0.882      0.608
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          32/90      6.64G      0.938     0.5515      1.111         72        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.864      0.842      0.885      0.606
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          33/90      6.63G     0.9302     0.5624      1.114         61        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.885      0.826      0.873      0.606
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          34/90      6.65G     0.9188     0.5506      1.106         64        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.914      0.841      0.899      0.622
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          35/90      6.65G      0.912     0.5508        1.1         35        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.936       0.83      0.895      0.622
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          36/90      6.63G       0.91     0.5357        1.1         68        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.923      0.853      0.905      0.634
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          37/90      6.65G     0.8995     0.5399      1.095         51        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.934      0.815      0.892      0.619
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          38/90      6.63G     0.9019     0.5207      1.091         40        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.938      0.831       0.89      0.612
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          39/90      6.65G     0.8917     0.5241      1.086         53        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.939      0.835       0.89      0.623
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          40/90      6.63G     0.8733     0.5167      1.081         56        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.897      0.845       0.89      0.622
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          41/90      6.64G     0.8736     0.5042      1.081         50        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.932      0.842      0.895      0.635
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          42/90      6.64G      0.861     0.5032      1.066         67        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.941      0.848      0.898      0.626
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          43/90      6.64G     0.8715     0.5021      1.081         56        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074       0.93      0.853      0.902      0.626
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          44/90      6.64G     0.8563     0.4945      1.068         66        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.917      0.866      0.912      0.641
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          45/90      6.64G     0.8636     0.4919      1.073         45        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.909      0.852      0.898      0.625
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          46/90      6.64G     0.8571     0.4828      1.065         62        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.896      0.865      0.906       0.64
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          47/90      6.65G     0.8421      0.477      1.062         48        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.928      0.864      0.908      0.649
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          48/90      6.62G      0.837     0.4826      1.061         46        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.874      0.879      0.905      0.636
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          49/90      6.66G     0.8342     0.4745      1.051         40        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.923      0.848      0.896      0.639
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          50/90      6.63G     0.8201     0.4642      1.045         49        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.909       0.85       0.91      0.642
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          51/90      6.64G     0.8193     0.4628      1.049         46        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.924      0.849       0.91      0.644
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          52/90      6.62G     0.8089     0.4606      1.045         67        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.901      0.857      0.897      0.636
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          53/90      6.64G     0.8098      0.457       1.04         69        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.919      0.863      0.913      0.652
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          54/90      6.64G     0.7997     0.4446      1.038         72        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.916      0.872      0.909      0.652
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          55/90      6.65G     0.8055     0.4512      1.041         41        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.932      0.864      0.918      0.658
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          56/90      6.65G     0.7969     0.4366      1.035         48        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.931      0.862      0.926      0.654
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          57/90      6.65G     0.7933     0.4457      1.032         69        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.889      0.855      0.905      0.645
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          58/90      6.64G     0.7815     0.4352      1.022         55        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.926      0.864      0.912       0.66
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          59/90      6.64G     0.7759     0.4267      1.024         56        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.944      0.851      0.911      0.656
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          60/90      6.62G     0.7624     0.4177      1.021         40        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.895      0.884      0.914      0.661
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          61/90      6.65G     0.7577     0.4158      1.017         61        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.928      0.863      0.918      0.662
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          62/90      6.65G     0.7615      0.414      1.022         41        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.923      0.863      0.912      0.657
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          63/90      6.64G     0.7559     0.4071      1.014         46        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074       0.91      0.869      0.907      0.653
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          64/90       6.7G     0.7544     0.4151      1.018         42        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.937      0.876      0.919      0.672
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          65/90      6.65G     0.7502     0.4082      1.016         39        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.946       0.86      0.917      0.668
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          66/90      6.63G     0.7362     0.4061      1.006         53        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.914      0.877      0.912      0.659
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          67/90      6.64G     0.7263     0.3963          1         51        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.895      0.892      0.914      0.665
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          68/90      6.65G     0.7293     0.3929     0.9952         50        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.927       0.89      0.926      0.673
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          69/90      6.64G      0.731     0.4002      1.005         64        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.934      0.876      0.924      0.672
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          70/90      6.64G     0.7255     0.3934      1.004         39        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074       0.94      0.871      0.923      0.671
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          71/90      6.64G     0.7114      0.383     0.9911         34        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.919       0.88      0.916      0.669
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          72/90      6.64G     0.7082     0.3815     0.9884         79        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074       0.93      0.876      0.922      0.671
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          73/90      6.65G     0.7001     0.3788     0.9835         44        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.927      0.867      0.918      0.666
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          74/90      6.64G     0.7039       0.38      0.987         57        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.925      0.874      0.913      0.668
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          75/90      6.64G     0.6985     0.3754     0.9815         47        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.934       0.87      0.919      0.676
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          76/90      6.64G     0.6888     0.3708     0.9856         35        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.864       0.92      0.675
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          77/90      6.64G     0.6826     0.3655     0.9739         60        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.945      0.862      0.914      0.676
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          78/90      6.64G      0.682     0.3659     0.9817         42        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.938      0.872      0.916      0.674
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          79/90      6.65G     0.6889     0.3657     0.9845         59        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.873      0.914      0.675
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          80/90      6.62G     0.6685     0.3578     0.9757         46        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.934      0.882      0.919      0.677
    Closing dataloader mosaic
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          81/90      6.63G     0.6514     0.3189     0.9571         65        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.942      0.876      0.915       0.67
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          82/90      6.65G     0.6339     0.3085     0.9433         25        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.938      0.877       0.92      0.678
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          83/90      6.65G      0.631     0.3057     0.9494         33        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.934      0.869      0.917      0.677
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          84/90      6.64G     0.6263     0.3037     0.9446         25        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.929      0.882      0.918      0.675
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          85/90      6.65G     0.6138     0.2992     0.9359         30        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.858      0.915      0.676
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          86/90      6.64G      0.613     0.2974     0.9418         40        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.937       0.86      0.918      0.677
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          87/90      6.65G     0.6084     0.2932     0.9348         36        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943       0.86      0.916      0.677
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          88/90      6.62G     0.5991      0.288     0.9302         34        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.855      0.916       0.68
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          89/90      6.64G     0.5982     0.2902     0.9292         52        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.931       0.86      0.914      0.679
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          90/90      6.62G     0.5868     0.2834     0.9233         24        640: 1
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.927      0.862      0.912      0.679
    
    90 epochs completed in 2.002 hours.
    Optimizer stripped from runs/detect/train8/weights/last.pt, 52.0MB
    Optimizer stripped from runs/detect/train8/weights/best.pt, 52.0MB
    
    Validating runs/detect/train8/weights/best.pt...
    Ultralytics YOLOv8.0.170 🚀 Python-3.9.18 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060, 12037MiB)
    Model summary (fused): 218 layers, 25843813 parameters, 0 gradients
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.855      0.916       0.68
                         0        647        765      0.981      0.949      0.988      0.796
                        11        647         25       0.82        0.6      0.686      0.471
                         3        647        580      0.915      0.852      0.941      0.754
                         4        647        314      0.981       0.92      0.967      0.726
                         6        647        149      0.938      0.809      0.883      0.535
                         8        647        645      0.984      0.877      0.953      0.732
                         9        647        596      0.981      0.978      0.992      0.746
    Speed: 0.3ms preprocess, 9.2ms inference, 0.0ms loss, 1.1ms postprocess per image
    


```python

#!zip -r /content/rede_treinada.zip /content/runs/
```


```python
#!rm -r /content/content/
#!rm -r /content/rede_treinada.zip
```

    rm: cannot remove '/content/rede_treinada.zip': No such file or directory
    


```python
# !unrar x /content/runs.part1.rar  /content/
```


```python
!ls {HOME}/runs/detect/train8
```

    args.yaml		results.png	      train_batch11362.jpg
    labels_correlogram.jpg	train_batch0.jpg      train_batch1.jpg
    labels.jpg		train_batch11360.jpg  train_batch2.jpg
    results.csv		train_batch11361.jpg  weights
    

# Validate custom model


```python
#Here, we are taking the model best weights and using them to validate the model. Similarly as before we are using
#CLI to do that, The only difference is our mode = val instead of train
#Validation Script is using test dataset that was not used beforew

%cd {HOME}

!yolo task=detect mode=val model={HOME}/runs/detect/train8/weights/best.pt data={HOME}/datasets/EEP_Detection-1/data.yaml
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    Ultralytics YOLOv8.0.170 🚀 Python-3.9.18 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060, 12042MiB)
    Model summary (fused): 218 layers, 25843813 parameters, 0 gradients
    [34m[1mval: [0mScanning /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-[0m
                     Class     Images  Instances      Box(P          R      mAP50  m
                       all        647       3074      0.943      0.856      0.916       0.68
                         0        647        765      0.981      0.949      0.988      0.795
                        11        647         25       0.82        0.6      0.686      0.471
                         3        647        580      0.915      0.852      0.941      0.754
                         4        647        314      0.985      0.924       0.97      0.728
                         6        647        149      0.938      0.809      0.883      0.532
                         8        647        645      0.984      0.877      0.953      0.734
                         9        647        596      0.981      0.978      0.992      0.748
    Speed: 0.5ms preprocess, 16.7ms inference, 0.0ms loss, 2.2ms postprocess per image
    Results saved to [1mruns/detect/val[0m
    

# **Displaying the Confusion Matrix**


```python
#Confusion matrix is the chart that shows how our model handles different classes
#92% of the time the model detected correctly that the person is wearing jacket, while 1% of the time we get the Bounding Box but
#the jacket is incorrectly classified as Eye wear, while 7% of the time when person is wearing the Jacket the model is unable to detect it
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/val/confusion_matrix_normalized.png', width=900)
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    




    
![png](main_1_files/main_1_34_1.png)
    



# **Training and Validation Loss**


```python
# Here is the graph of the training and validation loss
#box loss and class loss is important
# The behavior of the model is convincing the model is coverging, Training more will give better results
%cd {HOME}

Image(filename=f'{HOME}/runs/detect/train8/results.png', width=600)
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    




    
![png](main_1_files/main_1_36_1.png)
    




```python
#Model Prediction on validation batch. These image are not used strictly for training so it is always better to take a
#look and see how model is behaving
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/val/val_batch0_pred.jpg', width=600)
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    




    
![jpeg](main_1_files/main_1_37_1.jpg)
    



# **Inference with Custom Model**


```python
#Inference means a prediction that we can run on an image to detect the label,
# whether classification or of a bounding box or a segmentation
# Testing the Model on Test Dataset images
%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train8/weights/best.pt conf=0.25 source='datasets/EEP_Detection-1/test/images'
```

    /home/fabio/Documents/Github/Yolov8/Aula_3
    Ultralytics YOLOv8.0.170 🚀 Python-3.9.18 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 3060, 12042MiB)
    Model summary (fused): 218 layers, 25843813 parameters, 0 gradients
    
    image 1/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/000446_jpg.rf.ae946a1122b8eaac534a4954f07b3755.jpg: 640x640 2 0s, 2 8s, 18.1ms
    image 2/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_161946_jpg.rf.149c26cf563aa2ace0bdc099c4af8fef.jpg: 640x640 (no detections), 17.8ms
    image 3/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162015_jpg.rf.f71cba1429b26a3c95b6f8706ea69853.jpg: 640x640 1 4, 18.2ms
    image 4/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162036_jpg.rf.f6b8fdbdef55ce36cd4ad61549b2c1e2.jpg: 640x640 1 4, 19.1ms
    image 5/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162308_jpg.rf.32a4cbe5e0f3e1bc5159596a1bbb833d.jpg: 640x640 2 6s, 17.2ms
    image 6/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162435_jpg.rf.849775b3a423d9ac0ceb11741451085b.jpg: 640x640 2 6s, 17.9ms
    image 7/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162444_jpg.rf.cb838ace82c39fe5929854eae07780a5.jpg: 640x640 1 4, 2 6s, 17.9ms
    image 8/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/20220721_162455_jpg.rf.45eaff36534d95d15b3acced876f060b.jpg: 640x640 1 4, 2 6s, 1 9, 17.1ms
    image 9/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/40_jpg.rf.4cb6d0cb08cfcd10687d97e04c49d479.jpg: 640x640 1 6, 17.3ms
    image 10/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/56_jpg.rf.b8cf9732124949a41c62edc69f6e0959.jpg: 640x640 1 11, 1 6, 17.1ms
    image 11/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/5_jpg.rf.f369af6b71aeb9ed64b4b5abe6f00a09.jpg: 640x640 2 11s, 17.1ms
    image 12/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/65_jpg.rf.8f24e83e9784a158d23e14accf74f83c.jpg: 640x640 3 3s, 17.8ms
    image 13/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/69_jpg.rf.76ecc5fbd931818fec4745b5c1570107.jpg: 640x640 1 0, 2 3s, 1 9, 18.5ms
    image 14/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/AightOne0569_jpg.rf.21ec62bbeda72c5cd4a1838ec5e06211.jpg: 640x640 2 3s, 18.5ms
    image 15/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/AightOne0641_jpg.rf.4ebe50401d7d6d67f6c957af74742493.jpg: 640x640 2 3s, 17.7ms
    image 16/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/AightOne0748_jpg.rf.bc03c0360f277df8197a0673bbe854e0.jpg: 640x640 5 3s, 1 8, 17.4ms
    image 17/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Aitin1845_jpg.rf.1f3e3b9a2d3a4a843d62a4f3ad751709.jpg: 640x640 1 3, 17.0ms
    image 18/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Aitin2331_jpg.rf.6e8a711618b9b4dce517b238b55d6c45.jpg: 640x640 3 0s, 9 3s, 18.1ms
    image 19/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Aitin2428_jpg.rf.9d787311f25b9d12f388d963cfe49537.jpg: 640x640 1 0, 2 3s, 1 6, 3 8s, 17.5ms
    image 20/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Aitin3355_jpg.rf.b602f466f744b37d1c3ca642b85cb571.jpg: 640x640 5 0s, 18 3s, 1 6, 3 8s, 18.4ms
    image 21/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/IMG_1037_jpg.rf.3636d4f7f40062dc0cdd1ccfffdc0578.jpg: 640x640 1 3, 16.7ms
    image 22/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/IMG_1156_jpg.rf.4cc84ff4d7250a25684424efc827c946.jpg: 640x640 1 3, 17.6ms
    image 23/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/IMG_1243_jpg.rf.ebfb88a9b5bff281e0ebba3107ddf787.jpg: 640x640 1 0, 1 3, 16.9ms
    image 24/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/IMG_1355_jpg.rf.56428c1d0d8477a11cf0f4b1e3ee950b.jpg: 640x640 1 0, 1 3, 17.6ms
    image 25/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/IMG_1357_jpg.rf.e1ad74f4f32db4a06264d09d25be02c6.jpg: 640x640 2 0s, 1 3, 16.7ms
    image 26/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Nein5336_jpg.rf.f33ad8b22941dbadfbf9cf66b6d2d534.jpg: 640x640 1 3, 17.7ms
    image 27/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Nein5542_jpg.rf.f8c2f7439f14f2b6e03e5d4a8ccdf0be.jpg: 640x640 1 3, 16.6ms
    image 28/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0041_jpg.rf.d07efc3328ccd1dc2cd93a179bd52ead.jpg: 640x640 1 0, 19.0ms
    image 29/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0269_jpg.rf.a80acdeceef19875f4bd58b1c0076f6d.jpg: 640x640 1 0, 17.6ms
    image 30/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0284_jpg.rf.687af8070b3035d38ba64b4bac2f8f20.jpg: 640x640 1 0, 17.2ms
    image 31/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0550_jpg.rf.21e9c895309ae57f66d4cb1f09cb0572.jpg: 640x640 1 0, 18.1ms
    image 32/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0554_jpg.rf.ca7418c3df1b9a684a79c5251af2d9c2.jpg: 640x640 1 0, 18.4ms
    image 33/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Phiphtin0609_jpg.rf.998d7d49e05d87564a0397ba813dd845.jpg: 640x640 1 0, 18.6ms
    image 34/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin0001_jpg.rf.8559dbe0203d63c941f68878f31b9af5.jpg: 640x640 1 3, 16.8ms
    image 35/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin0304_jpg.rf.acdeb1ca58cd6aecd6956d73b9dc5847.jpg: 640x640 1 3, 17.1ms
    image 36/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin0397_jpg.rf.8bbe0e3c1b070e4fec25f5b5865de9ef.jpg: 640x640 1 3, 17.3ms
    image 37/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin0460_jpg.rf.8a68b95147cda0492ac6441d5719caa7.jpg: 640x640 1 3, 16.7ms
    image 38/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin0529_jpg.rf.73277c5dff01cdb6737b9fd17db67d2b.jpg: 640x640 1 3, 17.3ms
    image 39/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin1085_jpg.rf.4e40d6b7ea926f96c572a44e98de2f3d.jpg: 640x640 1 3, 16.7ms
    image 40/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin1256_jpg.rf.853676c9e0216ec9faf3b086c4580452.jpg: 640x640 1 3, 16.5ms
    image 41/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Photin1286_jpg.rf.c86b0aba8828d3154cd942674aad163c.jpg: 640x640 1 3, 17.1ms
    image 42/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Sikctin0468_jpg.rf.ff06f171a4d5511ff4932d85d0f04711.jpg: 640x640 6 0s, 2 8s, 18.2ms
    image 43/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0057_jpg.rf.dcde557a02b71ca553ffd7dcb93d3bc4.jpg: 640x640 1 0, 17.6ms
    image 44/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0064_jpg.rf.db1b67cd085257daed9350f249904bc9.jpg: 640x640 1 0, 16.4ms
    image 45/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0154_jpg.rf.2c19039c265d41c582b4cda4ad07306f.jpg: 640x640 1 0, 2 8s, 17.0ms
    image 46/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0215_jpg.rf.8a94c33e6511d1f4b2569b63cc619da2.jpg: 640x640 1 0, 2 8s, 17.9ms
    image 47/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0261_jpg.rf.d97cb3372c05d0ad3741fc23cf558278.jpg: 640x640 7 0s, 2 8s, 18.2ms
    image 48/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0315_jpg.rf.b0607dc0a3d99c5f0c4886334722289d.jpg: 640x640 1 0, 2 8s, 18.5ms
    image 49/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0642_jpg.rf.0090c84a9854c4d99022616c344a6d85.jpg: 640x640 1 0, 2 8s, 17.4ms
    image 50/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo0723_jpg.rf.e1f9657be6fdbb984af68469bffca9d9.jpg: 640x640 7 0s, 1 8, 16.2ms
    image 51/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1079_jpg.rf.176b09ebb908a2cb5d828181f5dd3218.jpg: 640x640 1 0, 2 8s, 16.7ms
    image 52/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1125_jpg.rf.ae99fadeafbe514c1005524ead682cc8.jpg: 640x640 1 0, 3 8s, 18.2ms
    image 53/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1169_jpg.rf.22f8d722017e0a78e884ef8f273868d4.jpg: 640x640 1 0, 2 8s, 17.7ms
    image 54/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1280_jpg.rf.2c78725a144aaef7d9b9772f3de7db90.jpg: 640x640 1 0, 2 8s, 16.7ms
    image 55/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1289_jpg.rf.df12b6b9fd6d35e668f100d24fa9c46c.jpg: 640x640 6 0s, 2 8s, 17.3ms
    image 56/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1307_jpg.rf.8f5e6d6c448246ff66665b2cb0572ac2.jpg: 640x640 7 0s, 2 8s, 16.7ms
    image 57/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1349_jpg.rf.a4c23c69c666b0e4e677f2a69508100d.jpg: 640x640 6 0s, 2 8s, 18.2ms
    image 58/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1668_jpg.rf.c04415ff551f5bb4a71d732164754284.jpg: 640x640 7 0s, 2 8s, 16.8ms
    image 59/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1669_jpg.rf.1dfae7078713b5d7427286176ef5d3ee.jpg: 640x640 7 0s, 2 8s, 17.2ms
    image 60/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1719_jpg.rf.b3647157d3b35cb6f2291d9f719386c8.jpg: 640x640 7 0s, 2 8s, 17.0ms
    image 61/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1800_jpg.rf.2ce8766d4e86ffae7ebcd94d32234091.jpg: 640x640 1 0, 1 8, 18.2ms
    image 62/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo1952_jpg.rf.dd53a3096f47df801334214f7249b26e.jpg: 640x640 6 0s, 2 8s, 18.0ms
    image 63/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo2406_jpg.rf.1393a6d7f8886b16fba7655acc090bfc.jpg: 640x640 6 0s, 2 8s, 17.3ms
    image 64/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo2461_jpg.rf.aca37aac946c0fdc32f087b85a31bda8.jpg: 640x640 6 0s, 2 8s, 17.8ms
    image 65/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo2657_jpg.rf.0056a04e391e1746e99444af2857d205.jpg: 640x640 6 0s, 1 8, 16.7ms
    image 66/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/SikctinTwo2881_jpg.rf.789b27b63f2353db7cffa7d16a5b5570.jpg: 640x640 6 0s, 1 8, 16.6ms
    image 67/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Tan0208_jpg.rf.320cc799cbedb73b7a03da3dc007c067.jpg: 640x640 1 3, 16.6ms
    image 68/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Tan0289_jpg.rf.806131773e0bfdeeb76d18f2a28ccbdc.jpg: 640x640 1 3, 17.4ms
    image 69/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Tan0617617_jpg.rf.44eeb18f2e6b2885975d7efc6c9f3bf6.jpg: 640x640 1 0, 4 3s, 17.0ms
    image 70/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Tan0816816_jpg.rf.dbe64aad457be38809154108b413b48d.jpg: 640x640 1 0, 4 3s, 17.8ms
    image 71/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Tan0914914_jpg.rf.b3238bf14b9d6a0fc11d485590467c12.jpg: 640x640 1 0, 4 3s, 17.2ms
    image 72/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0016_jpg.rf.1aec7c9707e37532a68d7dde0d662f28.jpg: 640x640 5 0s, 4 3s, 17.2ms
    image 73/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0287_jpg.rf.7b2d19843f53ce940e2c09311ffcc6a2.jpg: 640x640 2 0s, 2 3s, 17.5ms
    image 74/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0313_jpg.rf.31e2888d9ed68c40c47d3c74712492fa.jpg: 640x640 1 0, 1 3, 16.9ms
    image 75/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0664_jpg.rf.e0382cb37f091b0706e920554e872877.jpg: 640x640 2 0s, 2 3s, 17.8ms
    image 76/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0770_jpg.rf.a8ddda04d650d3bc0a5bab43d4fcf5df.jpg: 640x640 3 0s, 3 3s, 1 4, 17.9ms
    image 77/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0773_jpg.rf.02c1bf87988833b7b8654858d4ae9aa8.jpg: 640x640 3 0s, 3 3s, 1 4, 17.2ms
    image 78/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0889_jpg.rf.26cf252880797654cbcc98170607f714.jpg: 640x640 1 4, 18.1ms
    image 79/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0891_jpg.rf.94c00e08c8f07a4d6e824ca667bccbd1.jpg: 640x640 1 4, 16.8ms
    image 80/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Thavantin0989_jpg.rf.a6d5485150b9ca3a941c371b16b20a5f.jpg: 640x640 2 0s, 3 3s, 1 4, 18.1ms
    image 81/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/ThavantinDuo0185_jpg.rf.8da2958293ac5b3e11002c320dd28377.jpg: 640x640 5 0s, 4 3s, 17.9ms
    image 82/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/ThavantinDuo0570_jpg.rf.aa76891d0810b047afbf2168c1d60fd6.jpg: 640x640 3 0s, 2 3s, 17.0ms
    image 83/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/ThavantinDuo1237_jpg.rf.87f66759a495b44a7f4f1f5f6bb1fd7d.jpg: 640x640 2 0s, 4 3s, 1 6, 18.4ms
    image 84/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/TsirinTu0012_jpg.rf.0dcd75e9501efed9df88bb37983867dc.jpg: 640x640 6 3s, 16.1ms
    image 85/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Twalv0106_jpg.rf.cb404a11d05d39280af4f220cfc7b438.jpg: 640x640 2 3s, 17.3ms
    image 86/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Twalv0282_jpg.rf.1597dac1f7595cbec628aac7b09b58c2.jpg: 640x640 1 3, 17.2ms
    image 87/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Twalv0535_jpg.rf.e8fe97f1f701df13d343abf19c9fbe6e.jpg: 640x640 1 3, 16.7ms
    image 88/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Twalv0798_jpg.rf.e0f6e0f750826efa71e76d0bddc4c7ed.jpg: 640x640 4 3s, 17.9ms
    image 89/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/Twalv1203_jpg.rf.ee6fbcd52e95b69d7d9b92b20050eff1.jpg: 640x640 1 3, 17.1ms
    image 90/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/download--10-_jpg.rf.d1b4c0177062114a45eee85a6ec7ba8a.jpg: 640x640 1 3, 18.3ms
    image 91/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/download--2-_jpg.rf.e72a04b9447e07b1d3fb0fad0c965399.jpg: 640x640 2 0s, 1 3, 16.5ms
    image 92/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/download--22-_jpg.rf.abbe2e760fe7b83d428e184a94e475e9.jpg: 640x640 1 3, 16.8ms
    image 93/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/download--34-_jpg.rf.9bd74675c74752e26142a6f43575f7d9.jpg: 640x640 1 3, 17.0ms
    image 94/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/download--45-_jpg.rf.7a01abe712500fddb5364fe6c8b17743.jpg: 640x640 1 3, 17.2ms
    image 95/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/frame-078_jpg.rf.eaaed8174a3e2d3789a5521b18c75692.jpg: 640x640 1 0, 17.7ms
    image 96/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/frame-132_jpg.rf.4145ce155ed8e208a3c625e36836cb76.jpg: 640x640 1 0, 1 3, 17.1ms
    image 97/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_100_jpg.rf.cb7c438ef14d18d193a22a82d5be4ae7.jpg: 640x640 1 0, 1 3, 1 4, 1 8, 2 9s, 18.4ms
    image 98/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_102_jpg.rf.1326a15abf20a1fda54b0b111d6a0b46.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.6ms
    image 99/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_102_jpg.rf.e1eab37caaa444dd1d789253ec2906ce.jpg: 640x640 1 0, 1 4, 4 9s, 17.3ms
    image 100/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_103_jpg.rf.942fb055d109c9a14caec3e8627e8642.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.4ms
    image 101/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_109_jpg.rf.0780aa6477a27666cf7d4327ff4dfe41.jpg: 640x640 1 0, 1 3, 1 4, 17.6ms
    image 102/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_109_jpg.rf.42da9bd26056ea766d61dbe3deb27d64.jpg: 640x640 1 0, 1 4, 2 8s, 2 9s, 17.4ms
    image 103/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_109_jpg.rf.8b667dffb4175eb7047bddecc34bc29b.jpg: 640x640 1 0, 1 3, 1 4, 2 9s, 17.3ms
    image 104/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_109_jpg.rf.a4d65fcf94462dc82e7b02c68da14e12.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 9s, 16.7ms
    image 105/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_10_jpg.rf.8190931340569a39e12590f0069d7fa3.jpg: 640x640 1 0, 1 3, 1 4, 2 9s, 17.5ms
    image 106/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_110_jpg.rf.70edacd3038bc7fbd928df83ce12115e.jpg: 640x640 1 0, 1 3, 1 4, 1 8, 2 9s, 16.8ms
    image 107/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_111_jpg.rf.874bf0bae3dfb668569ef76372059f52.jpg: 640x640 1 0, 1 4, 18.2ms
    image 108/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_112_jpg.rf.6c909c4c3f7ce19816231233b5bba45d.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.2ms
    image 109/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_112_jpg.rf.7509f405f0c217e009156a503b7c65c3.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 2 9s, 17.3ms
    image 110/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_116_jpg.rf.f15f96132203ec1ddd4e1ca0c96bd694.jpg: 640x640 1 0, 1 4, 2 8s, 2 9s, 17.0ms
    image 111/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_117_jpg.rf.4c680ac4777e874809db218a3c3342ca.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 2 9s, 17.8ms
    image 112/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_118_jpg.rf.93672bfea9b27566e528b05895707825.jpg: 640x640 2 9s, 16.9ms
    image 113/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_118_jpg.rf.cb3a6f6c5f7434ceb1c4cd824e626a37.jpg: 640x640 1 0, 2 3s, 2 8s, 2 9s, 18.6ms
    image 114/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_128_jpg.rf.837f3ab853f35a3f2a7bc6465084ee94.jpg: 640x640 1 0, 2 3s, 2 8s, 2 9s, 17.6ms
    image 115/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_129_jpg.rf.a529a858be899cf28874cffca978e977.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 17.8ms
    image 116/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_12_jpg.rf.19977b574e8d4ec98a9391a277c89e3f.jpg: 640x640 1 3, 17.3ms
    image 117/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_12_jpg.rf.2d788b19197908df08f2c2cb9ff184a0.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 1 9, 16.9ms
    image 118/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_139_jpg.rf.5e89d76f6875fa7a5225f344a9efb8fa.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 2 9s, 17.0ms
    image 119/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_13_jpg.rf.ecad580767d3384b399ac857a5f5d501.jpg: 640x640 1 8, 17.0ms
    image 120/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_13_jpg.rf.f293063bc8a44dca429599a67146460f.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 16.9ms
    image 121/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_140_jpg.rf.07e082109717beda86e36cc26f9594de.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 17.6ms
    image 122/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_141_jpg.rf.221a2d49537c228167a867c2f6df790d.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 1 9, 16.6ms
    image 123/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_141_jpg.rf.9b2a15ba12df897f001505daf516af69.jpg: 640x640 1 0, 1 4, 2 9s, 17.3ms
    image 124/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_141_jpg.rf.e43721c2afa6236a1ff35888e0a90414.jpg: 640x640 1 3, 1 4, 17.4ms
    image 125/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_144_jpg.rf.fc0bda5c756caac2740491afef380cae.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 17.8ms
    image 126/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_145_jpg.rf.6e8a37295dbff2f2947f577895c9d948.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 17.5ms
    image 127/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_149_jpg.rf.5415ac22c7606e27c1fd3ab19828d664.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 3 9s, 18.1ms
    image 128/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_149_jpg.rf.7c3e36c7600375e7b172307de0ca0e04.jpg: 640x640 1 0, 1 3, 1 4, 1 8, 3 9s, 17.0ms
    image 129/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_14_jpg.rf.99b61f99972ec879e3f26322229bab28.jpg: 640x640 1 3, 1 4, 1 6, 1 8, 17.6ms
    image 130/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_150_jpg.rf.9749347278c3e83797f6dac5673934cd.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.1ms
    image 131/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_154_jpg.rf.b1aefa4fae6afda7fb241aa7448cb14b.jpg: 640x640 1 3, 1 4, 1 6, 2 9s, 17.4ms
    image 132/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_155_jpg.rf.cd0770bcdf28de5cfd8a34a6197bb8de.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.0ms
    image 133/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_155_jpg.rf.e363ec6c4204edc0cc8a06dd7b597fb6.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.6ms
    image 134/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_161_jpg.rf.b144a907465e9fe5b2b4a2a991965359.jpg: 640x640 1 0, 2 3s, 2 4s, 1 6, 2 8s, 2 9s, 16.7ms
    image 135/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_162_jpg.rf.56e0e9520677fbb9e162a38044c73ac7.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.1ms
    image 136/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_163_jpg.rf.40d89b4f2702b319adc7f36dadac65a5.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.8ms
    image 137/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_166_jpg.rf.0e55fc45f718108f0364ca3d2c1e1baf.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.7ms
    image 138/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_16_jpg.rf.0f00ef4f13452fa39b109c94ee32843d.jpg: 640x640 1 3, 1 4, 16.9ms
    image 139/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_16_jpg.rf.deb147c74d348ca3840ae7fdf072d4b7.jpg: 640x640 1 3, 1 4, 17.3ms
    image 140/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_170_jpg.rf.145ff27205d727fab8147744353ffe92.jpg: 640x640 1 4, 1 8, 2 9s, 18.0ms
    image 141/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_170_jpg.rf.4cf34a5b7ccaa5553582bf6b7fde8b1e.jpg: 640x640 1 0, 1 3, 1 4, 4 9s, 16.5ms
    image 142/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_171_jpg.rf.ca08a4fcc6871e964ed956530e930c84.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 18.2ms
    image 143/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_172_jpg.rf.43ec8750cdf9c1d7a884e7370e3d61d8.jpg: 640x640 1 4, 2 9s, 17.7ms
    image 144/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_172_jpg.rf.980664fc56e683fc4a6d56728e075a4a.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 17.3ms
    image 145/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_172_jpg.rf.bf1419d29883c7900cacdd014563a720.jpg: 640x640 1 0, 1 3, 2 8s, 2 9s, 16.8ms
    image 146/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_173_jpg.rf.20c20a2da795175517601238eaa5c193.jpg: 640x640 1 0, 2 3s, 1 4, 1 6, 2 8s, 2 9s, 17.0ms
    image 147/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_174_jpg.rf.f205d297d5ab4e02e2709f67a35b3032.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.4ms
    image 148/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_175_jpg.rf.82fb7b349c313e5a54e2dc9135605cbc.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.2ms
    image 149/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_176_jpg.rf.24aeebb67a10b43362786824076e7182.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.3ms
    image 150/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_179_jpg.rf.057182cda91057a051bd3f96d3f2f0b8.jpg: 640x640 1 3, 2 9s, 17.3ms
    image 151/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_179_jpg.rf.aa2e082b470f078beb162d994a73783f.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.1ms
    image 152/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_17_jpg.rf.383b1ea9f08d6a0c63bfcdb2f4bc8013.jpg: 640x640 2 0s, 1 3, 1 4, 1 8, 17.5ms
    image 153/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_17_jpg.rf.6a601029beb6038fcfaedd7b9c634109.jpg: 640x640 2 0s, 1 3, 1 4, 1 6, 1 8, 16.9ms
    image 154/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_181_jpg.rf.327d859172190ae19d0b7a6619622e35.jpg: 640x640 1 3, 2 9s, 17.2ms
    image 155/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_181_jpg.rf.6009328abcad6562527ab57c845dd9bd.jpg: 640x640 1 3, 2 9s, 17.0ms
    image 156/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_184_jpg.rf.db8debb41e39db5a6f087b0001032283.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.1ms
    image 157/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_185_jpg.rf.301ed9bb03873d9e802a1aed4d9a9e58.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.6ms
    image 158/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_185_jpg.rf.6daf8f53eaadf136f693dc364f53c434.jpg: 640x640 1 4, 2 9s, 18.1ms
    image 159/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_185_jpg.rf.cd42c91b50105124d3d862af106e3541.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.6ms
    image 160/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_187_jpg.rf.a892d5435c55e2b6868eede8ef27cf8c.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.4ms
    image 161/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_188_jpg.rf.a6644ddb23638fecda0f2a4339fb374b.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.1ms
    image 162/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_18_jpg.rf.0f1336c9de541e6689c131ac46f0dbba.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 18.2ms
    image 163/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_191_jpg.rf.10a883f000b8bb4c688375480e6f8f1c.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 16.9ms
    image 164/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_192_jpg.rf.53fbd282851b6798eabb4d46f136547c.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.8ms
    image 165/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_194_jpg.rf.a6eaccc5a8577ef8afb3f3a0a81bfb36.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.7ms
    image 166/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_195_jpg.rf.056d5e36707a9e712a2928ca2fc4d22e.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.8ms
    image 167/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_198_jpg.rf.a90580259396cd61e7cf5a25b91e8398.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.0ms
    image 168/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_202_jpg.rf.a8da380d2d7642848964641e56bfda3d.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 16.8ms
    image 169/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_20_jpg.rf.a0765e04df08d1eb8a0930fd89c887f4.jpg: 640x640 1 0, 1 4, 2 9s, 18.1ms
    image 170/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_210_jpg.rf.f566fad794e9d153db88d6d1b881dd41.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.2ms
    image 171/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_212_jpg.rf.417c5c423e0406bb96e57e7687f0839c.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.0ms
    image 172/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_212_jpg.rf.5641f309d1a678a323bb020a0b54635e.jpg: 640x640 2 9s, 17.2ms
    image 173/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_213_jpg.rf.8644744d7586ea51931f45a1e2c84bba.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.1ms
    image 174/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_215_jpg.rf.23b53191bc3828e62737472f1dbaf808.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 16.9ms
    image 175/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_216_jpg.rf.94a9fe8d2dfa051888b3a461c0accd85.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 2 9s, 18.2ms
    image 176/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_219_jpg.rf.0fc29acc7fc767a83ff370bc196f2336.jpg: 640x640 2 9s, 17.2ms
    image 177/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_219_jpg.rf.f700e3d0883b085d2ee61bd4ae9b1657.jpg: 640x640 2 9s, 18.1ms
    image 178/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_21_jpg.rf.24a8057b5a18cbb15d12d4c80d8ef631.jpg: 640x640 1 3, 2 8s, 16.8ms
    image 179/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_221_jpg.rf.6a80ea11dab623c5d386b98a08ebe2e9.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.6ms
    image 180/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_223_jpg.rf.9c2b605db2b0af80588b6959d77b3cd9.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.3ms
    image 181/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_223_jpg.rf.f03eb07e8f5bf70975b7dd4bb5631843.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.3ms
    image 182/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_224_jpg.rf.319615842fec633614f1d918a848d111.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 16.8ms
    image 183/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_224_jpg.rf.4abf8b010e70c1ace2d3a5e2432a5299.jpg: 640x640 1 3, 1 4, 2 8s, 2 9s, 17.2ms
    image 184/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_226_jpg.rf.0acaf862bafd4e8725ff7201a76543c0.jpg: 640x640 2 9s, 16.9ms
    image 185/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_228_jpg.rf.93a794ae8d72a8c5c9772f46e7adf2ce.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.1ms
    image 186/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_230_jpg.rf.02568056abaf67dfc7231c107c9dafa6.jpg: 640x640 2 9s, 17.5ms
    image 187/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_230_jpg.rf.bd2c7cdf1c16828f69814ceb76da0f55.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.8ms
    image 188/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_231_jpg.rf.e0efa05a945732985df05c6e3deb1039.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 18.0ms
    image 189/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_234_jpg.rf.6be194f090bca9f7e90399adeb83796e.jpg: 640x640 2 9s, 17.0ms
    image 190/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_234_jpg.rf.70a8d1a863a08ad90fa70805fdcd8501.jpg: 640x640 1 3, 2 9s, 17.5ms
    image 191/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_236_jpg.rf.a5195e70074c77c4dbe6c1ecb0fc43f1.jpg: 640x640 1 0, 17.7ms
    image 192/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_238_jpg.rf.b2555eca83383c2d0628ad638232f970.jpg: 640x640 1 3, 1 6, 2 8s, 2 9s, 17.6ms
    image 193/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_23_jpg.rf.1b4c7a7ae6561fc94a70b31cd2d4260d.jpg: 640x640 1 0, 1 4, 2 9s, 18.1ms
    image 194/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_23_jpg.rf.4c30b11b293375740f700e156cab7d35.jpg: 640x640 1 0, 1 4, 2 9s, 17.3ms
    image 195/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_23_jpg.rf.e5792b12ea882dced3cbc3e7dca428e9.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 2 9s, 17.3ms
    image 196/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_23_jpg.rf.fadcfe63aabbcb20443f66180760c7f0.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 18.5ms
    image 197/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_241_jpg.rf.05f5d9526d11265ac4b87618f06fe328.jpg: 640x640 1 0, 1 3, 1 4, 1 8, 2 9s, 17.3ms
    image 198/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_247_jpg.rf.2b1e4c2f21f6f8ff399aca3c0f18a74d.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.6ms
    image 199/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_250_jpg.rf.b8cfe8217a3344903977dd67abb84590.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.5ms
    image 200/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_256_jpg.rf.8609523ff35143943f2ab020ababec39.jpg: 640x640 1 3, 1 4, 1 6, 1 8, 2 9s, 16.9ms
    image 201/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_257_jpg.rf.25842bdbfe1a2f7874ef346b7fba7944.jpg: 640x640 1 3, 2 8s, 2 9s, 16.7ms
    image 202/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_258_jpg.rf.94e8a43d46a69a6b2ec0a13b4f9764a3.jpg: 640x640 1 3, 1 4, 1 6, 1 8, 2 9s, 17.9ms
    image 203/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_25_jpg.rf.0aef6f4f9dabf86f4ec79f99685223e1.jpg: 640x640 1 3, 1 4, 17.3ms
    image 204/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_25_jpg.rf.20123ac6095f502d80b375f0c2c90b9e.jpg: 640x640 1 3, 1 4, 1 8, 17.0ms
    image 205/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_262_jpg.rf.ba599470e0460cd5db38f64c98a4cf05.jpg: 640x640 1 3, 2 8s, 2 9s, 18.7ms
    image 206/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_266_jpg.rf.5bf0ad3835f194b2b1f722c2e34fce69.jpg: 640x640 1 3, 2 8s, 2 9s, 17.0ms
    image 207/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_26_jpg.rf.3907fe91f15f37dc89c31b8523fdd5db.jpg: 640x640 1 0, 1 4, 2 9s, 17.4ms
    image 208/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_26_jpg.rf.e8a9eadc9345a0ab5d41f85d93a646a8.jpg: 640x640 1 3, 1 4, 17.9ms
    image 209/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_27_jpg.rf.e0c38a4377d21a4de9bfb4e9e6fe0a23.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 16.9ms
    image 210/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_282_jpg.rf.5120fb3932b79cb51916fdae0aaff166.jpg: 640x640 1 4, 2 8s, 2 9s, 18.0ms
    image 211/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_283_jpg.rf.16aef3d5a959c09ef7211f671f69501f.jpg: 640x640 1 4, 2 8s, 2 9s, 17.9ms
    image 212/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_284_jpg.rf.902779013214a76104db6eea99759b2e.jpg: 640x640 1 4, 2 8s, 2 9s, 17.2ms
    image 213/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_289_jpg.rf.fb223af9332043b29bc4e94a49bc0ff9.jpg: 640x640 2 8s, 2 9s, 17.8ms
    image 214/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_294_jpg.rf.202e8a1f715ec28050025ded632a0237.jpg: 640x640 2 8s, 2 9s, 17.3ms
    image 215/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_294_jpg.rf.dd7dcfe13d9c358c313ed3b16c36cb98.jpg: 640x640 1 4, 2 8s, 2 9s, 16.7ms
    image 216/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_306_jpg.rf.7206760fd87906499125a610d054d831.jpg: 640x640 2 8s, 2 9s, 16.3ms
    image 217/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_307_jpg.rf.fe6abdb7ebc0dea8fc364c4f20da76b0.jpg: 640x640 1 4, 2 8s, 2 9s, 16.6ms
    image 218/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_309_jpg.rf.f19ec9d46c8bcf1d756c92631fb3dcea.jpg: 640x640 1 4, 2 8s, 2 9s, 17.9ms
    image 219/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_316_jpg.rf.5a836e628f87e688db03f78cf9d2e590.jpg: 640x640 1 4, 2 8s, 2 9s, 16.7ms
    image 220/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_31_jpg.rf.602ec4cd15a168889ce4e9443af6ffc6.jpg: 640x640 1 4, 17.1ms
    image 221/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_323_jpg.rf.357e4c122f343827ec043196d5b870f5.jpg: 640x640 2 8s, 2 9s, 17.5ms
    image 222/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_323_jpg.rf.d8470e247f7148ddf7709363f1d4b9af.jpg: 640x640 2 8s, 2 9s, 18.6ms
    image 223/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_33_jpg.rf.4a911f9cf873eec1c5a84bc73f918d7d.jpg: 640x640 1 0, 1 3, 1 8, 2 9s, 17.3ms
    image 224/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_33_jpg.rf.83e40b5391f3672a5d6e3643a98dbb9a.jpg: 640x640 1 0, 1 4, 2 8s, 18.2ms
    image 225/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_33_jpg.rf.df14c802da90e2b0b1f3fa8be776685b.jpg: 640x640 1 0, 1 4, 2 8s, 18.4ms
    image 226/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_34_jpg.rf.082f7b87ed3aad376776d02671455810.jpg: 640x640 1 3, 1 4, 1 6, 18.2ms
    image 227/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_34_jpg.rf.5118f3782e48e3a0531727eadb1bec4a.jpg: 640x640 1 3, 2 8s, 17.1ms
    image 228/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_34_jpg.rf.8f31a5a47c800b85cb42ee48ef6296a0.jpg: 640x640 2 8s, 16.8ms
    image 229/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_34_jpg.rf.b7cc3917b2fe376ae02792153f0dfdbb.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 1 8, 17.5ms
    image 230/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_34_jpg.rf.bedaf3eda181968b7f6c8c6e95054cd3.jpg: 640x640 1 3, 2 8s, 17.2ms
    image 231/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_35_jpg.rf.be2eed2fa6716fedbd1aeaa1aaa48ff7.jpg: 640x640 1 0, 1 4, 18.4ms
    image 232/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_35_jpg.rf.f029980ddae4dd040bdbc6fc4fecc64b.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 16.9ms
    image 233/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_35_jpg.rf.f02a0a0df19037e75997ec2d9adf0c94.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.5ms
    image 234/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_36_jpg.rf.5568a3108af31203665a4491d686ba9b.jpg: 640x640 1 3, 1 4, 1 8, 18.0ms
    image 235/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_36_jpg.rf.cab31203f14693f01e862b0272f7b9aa.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 16.7ms
    image 236/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_36_jpg.rf.ffd91cfb65fc2772324e0bc39c97285e.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 17.3ms
    image 237/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_37_jpg.rf.a98221bbbfd2018485c26593dea191b1.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 16.8ms
    image 238/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_40_jpg.rf.df3882666b2b43464d2b62eb022b677e.jpg: 640x640 2 0s, 1 3, 1 4, 1 6, 2 8s, 17.7ms
    image 239/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_41_jpg.rf.a6ab2458d540c2f9eab90394bb863470.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 17.3ms
    image 240/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_41_jpg.rf.c031ccbed329fd0471e3a05fb6752962.jpg: 640x640 1 4, 3 9s, 17.2ms
    image 241/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_42_jpg.rf.0fae595271c73978739e5fd8bb442e81.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 16.9ms
    image 242/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_43_jpg.rf.04148fb5e85434c84b835b6f11db9cc0.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 18.0ms
    image 243/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_43_jpg.rf.403575c635c596de8413864b1f3145a1.jpg: 640x640 1 0, 1 4, 16.9ms
    image 244/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_43_jpg.rf.deec78900788e290eac87941ee46f6cb.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.0ms
    image 245/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_43_jpg.rf.f2958078ca4d33a24e51a29e8a0e474d.jpg: 640x640 2 0s, 1 3, 1 4, 1 6, 2 8s, 17.3ms
    image 246/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_44_jpg.rf.938bba1f79171b36c35c0f8b9615e1a0.jpg: 640x640 1 0, 1 3, 1 4, 2 6s, 2 8s, 17.2ms
    image 247/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_44_jpg.rf.b5a9beee8c128a1a7dfe5041fe2b8925.jpg: 640x640 1 0, 1 3, 2 8s, 17.4ms
    image 248/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_46_jpg.rf.a1fb42d869a381a75e91877b394e4c81.jpg: 640x640 1 0, 1 3, 1 4, 2 8s, 2 9s, 17.6ms
    image 249/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_47_jpg.rf.fc5c976a844745ef1caaeb5682110c2b.jpg: 640x640 1 0, 1 4, 18.0ms
    image 250/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_49_jpg.rf.3f2a473673789aa9c0359113ea7cae9b.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 1 9, 17.1ms
    image 251/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_50_jpg.rf.2bf5d26c07ae19ae80254bd1347f6da9.jpg: 640x640 1 0, 1 3, 1 4, 2 9s, 16.8ms
    image 252/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_51_jpg.rf.05ece893f44a913526f5bc15fa2fa0ff.jpg: 640x640 1 4, 2 9s, 18.0ms
    image 253/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_52_jpg.rf.a8e8b161f411faeb4e67b51f0655f933.jpg: 640x640 1 0, 1 3, 1 4, 2 9s, 17.5ms
    image 254/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_54_jpg.rf.c7098a5010af834dfe45b36017adeae4.jpg: 640x640 1 0, 1 4, 17.5ms
    image 255/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_55_jpg.rf.43c4d041b4f6fa0693bb212fc1dc5658.jpg: 640x640 1 0, 1 4, 2 9s, 17.8ms
    image 256/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_55_jpg.rf.4f92cdf8dc8d81477cb8b56a9f08a59e.jpg: 640x640 1 0, 1 4, 2 9s, 17.9ms
    image 257/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_55_jpg.rf.a736ca115bf42ac169a28db4da32d1df.jpg: 640x640 1 4, 2 9s, 16.8ms
    image 258/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_56_jpg.rf.317728f32766481d42e0ef07b5e5d32f.jpg: 640x640 1 0, 1 4, 18.2ms
    image 259/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_58_jpg.rf.c72cba4959a1bbbd3393e9c431b1c92a.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.9ms
    image 260/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_60_jpg.rf.f040052473aa8007432e71caa43188b9.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.1ms
    image 261/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_69_jpg.rf.199f9b1e4a9b74249ae9953aa3d84a79.jpg: 640x640 1 0, 2 3s, 1 4, 1 6, 2 8s, 2 9s, 18.4ms
    image 262/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_70_jpg.rf.347ad74136503ef6a1bafa30a70b785b.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.4ms
    image 263/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_73_jpg.rf.7549d272a786c157db21c23dfa861796.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.0ms
    image 264/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_77_jpg.rf.3fb8a2521645bc7e5be4fe45fd7868e1.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.2ms
    image 265/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_78_jpg.rf.81d52bd176bb8734a47424f4de3afbfd.jpg: 640x640 1 0, 2 8s, 2 9s, 17.1ms
    image 266/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_7_jpg.rf.0692685da7d2c01df41aa5ce79f8af78.jpg: 640x640 1 3, 1 4, 1 6, 2 8s, 17.3ms
    image 267/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_80_jpg.rf.bd26407f0141f9dbea102d6faedbb559.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.3ms
    image 268/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_83_jpg.rf.8afcf42505e1207cbf15584dc52225fa.jpg: 640x640 1 0, 2 3s, 1 4, 1 6, 2 8s, 2 9s, 18.3ms
    image 269/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_84_jpg.rf.bc4adb9f66f2fd92d38624d312cca563.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 18.1ms
    image 270/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_89_jpg.rf.911c6d1cbac23aa6acb4ae5186c92af0.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 16.7ms
    image 271/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_90_jpg.rf.451ee5c91e2a67405502865d389ab9b9.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 19.3ms
    image 272/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_91_jpg.rf.4275e28c8eb45b7914012a71d6afb36c.jpg: 640x640 1 4, 2 9s, 16.6ms
    image 273/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_93_jpg.rf.08278bccc7176a6918b8cea9a70f11db.jpg: 640x640 1 4, 2 9s, 16.7ms
    image 274/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_93_jpg.rf.49f83f0b5bf0036c35aafb029af57218.jpg: 640x640 1 0, 1 4, 1 9, 17.8ms
    image 275/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_94_jpg.rf.2c83f6dc1b89fe09ffcbca3fd911db98.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.6ms
    image 276/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_94_jpg.rf.a180b61614ef0dd63d9d291383eeaf83.jpg: 640x640 1 4, 17.8ms
    image 277/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_96_jpg.rf.ee7c06185ecad69654080a41b72ba655.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.3ms
    image 278/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_97_jpg.rf.5a5d756d4dad44bf92f3e0fc8a89f59b.jpg: 640x640 1 0, 1 3, 1 4, 1 6, 2 8s, 2 9s, 17.3ms
    image 279/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/image_99_jpg.rf.5c5c21ed09690414147e727677aca69a.jpg: 640x640 1 4, 2 9s, 18.4ms
    image 280/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T011718_jpg.rf.7bc04c8533162f91fd23014dd4bd9d0e.jpg: 640x640 1 11, 16.8ms
    image 281/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T012941_jpg.rf.ec21237b6dcc91c865f6feb6e22a36f3.jpg: 640x640 1 0, 1 3, 17.6ms
    image 282/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013055_jpg.rf.bd854042eca2c9be2f574990586ec0b1.jpg: 640x640 1 0, 17.2ms
    image 283/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013058_jpg.rf.d6dc8c239c7747817ca21fec416d3af1.jpg: 640x640 2 0s, 1 8, 17.7ms
    image 284/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013139_jpg.rf.5db8e3b2ca0f696b9343b2500518c148.jpg: 640x640 1 8, 17.6ms
    image 285/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013441_jpg.rf.7d59317df2ccbc7818afdcc3d9ba5792.jpg: 640x640 1 0, 2 8s, 17.6ms
    image 286/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013524_jpg.rf.0104ae285452217e95d76fcc76dd301e.jpg: 640x640 1 3, 17.2ms
    image 287/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013634_jpg.rf.2705a666775f7741bdadcec6d43c5330.jpg: 640x640 1 8, 18.5ms
    image 288/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013650_jpg.rf.0f983b810cbc2c0fa4dfad91de4b6d3a.jpg: 640x640 2 8s, 17.0ms
    image 289/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013653_jpg.rf.1c4b0ef39e6063d91ed46f8bf8ad5a95.jpg: 640x640 3 8s, 17.1ms
    image 290/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013832_jpg.rf.921266bee258360a8f04b707bece22f9.jpg: 640x640 (no detections), 16.9ms
    image 291/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013853_jpg.rf.cd6c2ccd9049eb5ebca5ded305bff659.jpg: 640x640 2 8s, 17.0ms
    image 292/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013931_jpg.rf.b216827dfabebf6ce557c1327cf74ef9.jpg: 640x640 1 8, 17.7ms
    image 293/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T013937_jpg.rf.972d78922acbfc4ef9e4d27d74a26e4d.jpg: 640x640 1 8, 19.2ms
    image 294/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T014326_jpg.rf.9a33ced81adcb8e2cba71207783cfe33.jpg: 640x640 2 3s, 17.2ms
    image 295/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T014519_jpg.rf.301ad72413ae5bfa9e4f17c0800a4fe0.jpg: 640x640 2 3s, 17.0ms
    image 296/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images---2022-07-04T014957_jpg.rf.8beceb42c382f615463d55230ae1ee23.jpg: 640x640 3 3s, 18.3ms
    image 297/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images--1-_jpg.rf.3b9ebe54c52f50e739a2619977ee1a42.jpg: 640x640 1 0, 17.2ms
    image 298/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/images--16-_jpg.rf.a42de788b62bc58b64111ef3c97da9d9.jpg: 640x640 1 0, 1 11, 17.2ms
    image 299/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs3054_jpg.rf.14a4005da31726955a3d59166f0a9735.jpg: 640x640 4 0s, 17.1ms
    image 300/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs3131_jpg.rf.6654e54d28a50cb03c3cd0e247c94070.jpg: 640x640 1 0, 17.0ms
    image 301/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs3467_jpg.rf.f7e57554c502bf240e8b4f3d8346f0d1.jpg: 640x640 1 0, 17.6ms
    image 302/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs4103_jpg.rf.0ffc12a4078aaa11b006b9b655e4a1cd.jpg: 640x640 1 0, 18.8ms
    image 303/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs6053_jpg.rf.d803e8d2da6d6c3685cd661aafad81ef.jpg: 640x640 1 0, 17.5ms
    image 304/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs6337_jpg.rf.062442b61c8bdb80333001c288bda809.jpg: 640x640 (no detections), 17.9ms
    image 305/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs6374_jpg.rf.33de5624956d2429140b5721e12a56ba.jpg: 640x640 1 0, 1 11, 17.5ms
    image 306/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs6751_jpg.rf.56ef58bcb20eadcbc631f8f4e1a39572.jpg: 640x640 5 4s, 16.7ms
    image 307/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs6888_jpg.rf.ff48dbf24b2dd00d26621817e4a68ed5.jpg: 640x640 1 0, 1 11, 18.8ms
    image 308/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs7037_jpg.rf.1a7c2ad8d3e2ac077ccf3d9392c053c5.jpg: 640x640 1 6, 17.3ms
    image 309/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/phifs7038_jpg.rf.7a9ac4e38b1139b9a9bec028c3c20568.jpg: 640x640 1 6, 16.8ms
    image 310/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_12_jpg.rf.778bfef16e7cf4e72c333a3b6a73f7d8.jpg: 640x640 1 11, 2 6s, 17.6ms
    image 311/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_20_jpg.rf.1a77ea22d49ca4489b199d2eb9d57542.jpg: 640x640 1 11, 1 6, 16.7ms
    image 312/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_48_jpg.rf.d3b6731deaaef29413cf7685d222af90.jpg: 640x640 1 11, 17.9ms
    image 313/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_56_jpg.rf.41578c7f97e2e42ac8f7e8184f83ece9.jpg: 640x640 1 11, 1 6, 16.9ms
    image 314/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_61_jpg.rf.b32a48a4cc6aba860dc07e515cb52c95.jpg: 640x640 1 11, 1 3, 1 6, 16.7ms
    image 315/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_65_jpg.rf.fe91021afbc4789b5f156d65119c05e8.jpg: 640x640 2 11s, 1 6, 1 8, 17.4ms
    image 316/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_8_jpeg_jpg.rf.4d8ff3401f72696515cbab1d3e627b1d.jpg: 640x640 1 11, 1 6, 17.6ms
    image 317/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/shield_99_jpeg_jpg.rf.ae1f376d71f89ed92e531cc61fa3f627.jpg: 640x640 1 11, 17.7ms
    image 318/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/sic5242_jpg.rf.5a218efba0e6ba2ad5282afa39728eb3.jpg: 640x640 3 0s, 2 3s, 16.2ms
    image 319/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/sird1545_jpg.rf.6993b7b635d5c7c78e9573cde2debdd4.jpg: 640x640 4 0s, 1 4, 18.9ms
    image 320/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/sird1561_jpg.rf.19763c9ae90198a46dceb059fd997ed4.jpg: 640x640 3 0s, 18.5ms
    image 321/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/sird3388_jpg.rf.4af99a52e288dbc58630aba1d530ec01.jpg: 640x640 2 0s, 18.2ms
    image 322/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/sird3486_jpg.rf.dbddb63ae3650082762e74c334f0b305.jpg: 640x640 3 0s, 17.6ms
    image 323/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/thumb0334_jpg.rf.072f032226ebcc810bd60eef5eaeb8ca.jpg: 640x640 1 0, 1 3, 16.6ms
    image 324/324 /home/fabio/Documents/Github/Yolov8/Aula_3/datasets/EEP_Detection-1/test/images/thumb2446_jpg.rf.99b76a9455fd79bfc3885d88d1fc03e5.jpg: 640x640 1 0, 1 3, 16.9ms
    Speed: 2.3ms preprocess, 17.4ms inference, 2.4ms postprocess per image at shape (1, 3, 640, 640)
    Results saved to [1mruns/detect/predict[0m
    


```python
import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict/*.jpg')[:5]:
      display(Image(filename=image_path, width=600))
      print("\n")
```


    
![jpeg](main_1_files/main_1_40_0.jpg)
    


    
    
    


    
![jpeg](main_1_files/main_1_40_2.jpg)
    


    
    
    


    
![jpeg](main_1_files/main_1_40_4.jpg)
    


    
    
    


    
![jpeg](main_1_files/main_1_40_6.jpg)
    


    
    
    


    
![jpeg](main_1_files/main_1_40_8.jpg)
    


    
    
    
