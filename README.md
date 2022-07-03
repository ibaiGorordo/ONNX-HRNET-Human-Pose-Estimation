# ONNX HRNET 2D Human Pose Estimation (WIP)
 Python scripts for performing 2D human pose estimation using the HRNET (HRNET, Lite-HRNet...) family models in ONNX.


![!ONNX HRNET 2D Human Pose Estimation](https://github.com/ibaiGorordo/ONNX-HRNET-Human-Pose-Estimation/blob/main/doc/img/output.jpg)
*Original image: https://commons.wikimedia.org/wiki/File:Bull-Riding2-Szmurlo.jpg*

# Important
- The repository is still in progress, I need to add the person detection to work with multiple people in the image.
- For the multiperson examples, it might be more efficient to collect all the image crops and pass them together to the models that accept multiple image batches (Nxheightxwidth). I do it separately for simplicity.

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
- HRNET: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/271_HRNet [MIT License](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/LICENSE)
- Lite HRNet: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/268_Lite-HRNet [Apache 2.0 License](https://github.com/HRNet/Lite-HRNet/blob/hrnet/LICENSE)
 
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

# References:
* HRNET: https://github.com/HRNet/HRNet-Human-Pose-Estimation
* Lite HRNet: https://github.com/HRNet/Lite-HRNet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* HRNET Original paper: https://arxiv.org/abs/1902.09212
* Lite HRNET Original paper: https://arxiv.org/abs/2104.06403
