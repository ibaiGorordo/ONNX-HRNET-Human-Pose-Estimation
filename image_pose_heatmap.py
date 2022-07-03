import cv2
from imread_from_url import imread_from_url

from HRNET import HRNET, ModelType

# Initialize inference model
model_path = "models/hrnet_coco_w48_Nx384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_threshold=0.0)

# Read image
img_url = "https://upload.wikimedia.org/wikipedia/commons/5/53/People_of_Tibet54.jpg"
img = imread_from_url(img_url)

# Perform the inference in the image
total_heatmap, peaks = hrnet(img)

# Draw Model Output
output_img = hrnet.draw_all(img)
cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", output_img)
cv2.imwrite("doc/img/output.jpg", output_img)
cv2.waitKey(0)