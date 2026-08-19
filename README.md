# RoboOcc
A 3d spatial query based robot vision occ. The training data is collected on Nvidia Isaaclab simulation platform.  
It's a demo project and it's not open source now.
# IsaacLab 
A simulation scene is created to collect data, the robot(ANYMAL_C) is controlled by keyboard "W"/"S"/"A"/"D" to walk forward/backward/turn left/turn right.   
A stereo camera is mounted on the top of the robot to get sensor data. A colored picture & depth RGB is acquired with 2Hz, and point cloud & occ label is calculate post process.  
<img src="isaaclab_1.png" alt="演示截图" style="width:50%"/><img src="isaaclab_2.png" alt="演示截图" style="width:50%"/>  
<img src="isaaclab_3.png" alt="演示截图" style="width:50%"/><img src="isaaclab_4.png" alt="演示截图" style="width:50%"/>
# Prediction
Bellow is the predicted rersult. Python test.py to run the inference.
![demo](./demo_gif.gif)
# Quickstart
The code has been tested both on Windows11 and Ubuntu 22.04. An anaconda virtual env is recommaned.  
```bash  
conda create -n roboocc python=3.10    
conda activate roboocc    
``` 
And install dependencies:  
pytorch >=2.1   
open3d  >=0.16.0  
opencv-python >= 4.7  
Download the model from huggingface.  
```bash  
https://huggingface.co/DolphinLi888/RoboOcc/tree/main
```  
# Inference
python test.y






