import cv2
from imread_from_url import imread_from_url

from HRNET import HRNET, ModelType, PersonDetector

# Initialize Pose Estimation model
model_path = "models/hrnet_coco_w48_Nx384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.5)

# Initialize Person Detection model
person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path)

# Read image
img_url = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Flickr_-_The_U.S._Army_-_%27cavalry_charge%27.jpg"
img = imread_from_url(img_url)

# Detect Objects in the image
detections = person_detector(img)

# Estimate the pose in the image
total_heatmap, peaks = hrnet(img, detections)

# Draw Model Output
output_img = hrnet.draw_all(img, mask_alpha=0.55)

# Draw detections
# output_img = person_detector.draw_detections(output_img)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", output_img)
cv2.imwrite("doc/img/output.jpg", output_img)
cv2.waitKey(0)
