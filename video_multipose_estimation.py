import cv2
import pafy

from HRNET import HRNET, PersonDetector
from HRNET.utils import ModelType, filter_person_detections

# # Initialize video
# cap = cv2.VideoCapture("input.avi")

videoUrl = 'https://youtu.be/HI-BMpNByo0'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 720))

# Initialize Pose Estimation model
model_path = "models/hrnet_coco_w48_384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.3)

# Initialize Person Detection model
person_detector_path = "models/yolov5s6.onnx"
person_detector = PersonDetector(person_detector_path, conf_thres=0.3)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
frame_num = 0
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()

        # Skip the first {start_time} seconds
        if frame_num < start_time * 30:
            frame_num += 1
            continue

        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Detect People in the image
    detections = person_detector(frame)
    ret, person_detections = filter_person_detections(detections)
    person_detector.boxes, person_detector.scores, person_detector.class_ids = person_detections

    if ret:

        # Estimate the pose in the image
        total_heatmap, peaks = hrnet(frame, person_detections)

        # Draw Model Output
        frame = hrnet.draw_pose(frame)
        frame = person_detector.draw_detections(frame, mask_alpha=0.15)

    cv2.imshow("Model Output", frame)
    # out.write(frame)

# out.release()
