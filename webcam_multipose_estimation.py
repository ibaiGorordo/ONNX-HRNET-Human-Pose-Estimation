import cv2

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize Pose Estimation model
model_path = "models/hrnet_coco_w48_384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.5)

# Initialize Person Detection model
person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect People in the image
    detections = person_detector(frame)
    ret, person_detections = filter_person_detections(detections)

    # Estimate the pose in the image
    total_heatmap, peaks = hrnet(frame, person_detections)

    if ret:

        # Draw Model Output
        output_img = hrnet.draw_pose(frame)
        cv2.imshow("Model Output", output_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
