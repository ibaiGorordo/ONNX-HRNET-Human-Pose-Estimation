import time
import cv2
import numpy as np
import onnxruntime

from .utils import *
from yolov6-inference import YOLOv6


class HRNET:

    def __init__(self, path, model_type, conf_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.model_type = model_type

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.update(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def update(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.total_heatmap, self.peaks = self.process_output(outputs)

        return self.total_heatmap, self.peaks

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, heatmaps):
        total_heatmap = heatmaps.sum(axis=1)[0]

        map_h, map_w = heatmaps.shape[2:]

        # Find the maximum value in each of the heatmaps and its location
        max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
        peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
                          for heatmap in heatmaps[0, ...]])
        peaks[max_vals < self.conf_threshold] = np.array([-1, -1])

        # Scale peaks to the image size
        peaks = peaks * np.array([self.img_height / map_h,
                                  self.img_width / map_w])

        return total_heatmap, peaks

    def draw_pose(self, image):
        return draw_skeleton(image, self.peaks, self.model_type)

    def draw_heatmap(self, image, mask_alpha=0.4):
        return draw_heatmap(image, self.total_heatmap, mask_alpha)

    def draw_all(self, image, mask_alpha=0.4):
        return self.draw_pose(self.draw_heatmap(image, mask_alpha))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    # Initialize model
    model_path = "../models/hrnet_coco_w48_384x288.onnx"
    model_type = ModelType.COCO
    hrnet = HRNET(model_path, model_type, conf_threshold=0.6)

    img = imread_from_url(
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/Bull-Riding2-Szmurlo.jpg")

    # Perform the inference in the image
    total_heatmap, peaks = hrnet(img)

    # Draw model output
    output_img = hrnet.draw_all(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_img)
    cv2.waitKey(0)
