# ONNX HRNET 2D Human Pose Estimation
 Python scripts for performing 2D human pose estimation using the HRNET family models (HRNET, Lite-HRNet) in ONNX.


![!ONNX HRNET 2D Human Pose Estimation](https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/blob/main/doc/img/output.jpg)
*Original image: https://en.wikipedia.org/wiki/File:Flickr_-_The_U.S._Army_-_%27cavalry_charge%27.jpg*

# Important
- For the multiperson examples, it might be more efficient to collect all the image crops and pass them together to the models that accept multiple image batches (Nxheightxwidth). I do it separately for simplicity.
- There are more efficient models to perform multi pose estimation, the approach presented here is not optimal.

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation.git --recursive
cd ONNX-HRNET-Human-Pose-Estimation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# ONNX model 
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309). Download the models from the link below and save them into the **[models](https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/tree/main/models)** folder"
- **HRNET** [[MIT License](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/LICENSE)]: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/271_HRNet 
- **Lite HRNet** [[Apache 2.0 License](https://github.com/HRNet/Lite-HRNet/blob/hrnet/LICENSE)]: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/268_Lite-HRNet 
 
## YOLOv5 or YOLOv6 ONNX models
For the multiperson examples, both [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOv6](https://github.com/meituan/YOLOv6) models can be used. You can convert the original models to ONNX using the Google Colab repositories linked below, and save the converted onnx models in to the **[models](https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/tree/main/models)** folder.
- **YOLOv5 ONNX conversion:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing)
- **YOLOv6 ONNX conversion:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pke1ffMeI2dXkIAbzp6IHWdQ0u8S6I0n?usp=sharing)

# Examples

 * **Image Single Pose Estimation**:
 ```
 python image_singlepose_estimation.py
 ```
 
  * **Image Multi Pose Estimation**:
 ```
 python image_multipose_estimation.py
 ```
 
 * **Image Single Pose Heatmap**:
 ```
 python image_singlepose_heatmap.py
 ```
 
 * **Webcam Multi Pose Estimation**:
 ```
 python webcam_multipose_estimation.py
 ``` 
 
 * **Video Multi Pose Estimation**:
 ```
 python video_multipose_estimation.py
 ``` 
 ![!HRNET Video Pose Estimation](https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/blob/main/doc/img/hrnet_video.gif)
 
 *Original video: https://youtu.be/HI-BMpNByo0*
  
# References:
* HRNET: https://github.com/HRNet/HRNet-Human-Pose-Estimation
* Lite HRNet: https://github.com/HRNet/Lite-HRNet
* YOLOv5: https://github.com/ultralytics/yolov5
* YOLOv6: https://github.com/meituan/YOLOv6
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* HRNET Original paper: https://arxiv.org/abs/1902.09212
* Lite HRNET Original paper: https://arxiv.org/abs/2104.06403
