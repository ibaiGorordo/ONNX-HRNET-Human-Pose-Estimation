import enum
import cv2
import numpy as np

# import matplotlib.pyplot as plt

coco_colors = [(255, 0, 127), (254, 37, 103), (251, 77, 77), (248, 115, 51),
               (242, 149, 25), (235, 180, 0), (227, 205, 24), (217, 226, 50),
               (206, 242, 76), (193, 251, 102), (179, 254, 128), (165, 251, 152),
               (149, 242, 178), (132, 226, 204), (115, 205, 230), (96, 178, 255),
               (78, 149, 255), (59, 115, 255), (39, 77, 255), (18, 37, 255), (0, 0, 255)]

coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                 [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                 [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                 [3, 5], [4, 6]]

mpii_skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                 [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                 [13, 14], [14, 15]]

mpii_colors = [(255, 0, 127), (253, 49, 95), (250, 97, 63), (243, 142, 31),
               (235, 180, 0), (224, 212, 32), (211, 236, 64), (196, 250, 96),
               (179, 254, 128), (161, 249, 160), (140, 234, 192), (119, 210, 224),
               (96, 178, 255), (72, 139, 255), (48, 95, 255), (23, 46, 255), (0, 0, 255)]


# cmap = plt.get_cmap('rainbow')
# colors = [cmap(i) for i in np.linspace(0, 1, len(mpii_skeleton) + 2)]
# colors_cv = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors]
# print(colors_cv)


class ModelType(enum.Enum):
    COCO = 1
    MPII = 2


def filter_person_detections(detections):
    boxes, scores, class_ids = detections

    boxes = boxes[class_ids == 0]
    scores = scores[class_ids == 0]
    class_ids = class_ids[class_ids == 0]

    return len(scores)>0, [boxes, scores, class_ids]


def get_vis_info(model_type):
    if model_type == ModelType.COCO:
        return coco_skeleton, coco_colors
    elif model_type == ModelType.MPII:
        return mpii_skeleton, mpii_colors
    else:
        raise ValueError("Unknown model type")

def valid_point(point):
    return point[0] >= 0 and point[1] >= 0

def draw_skeletons(img, keypoints, modeltype):
    output_img = img.copy()

    if type(keypoints) != list:
        return draw_skeleton(output_img, keypoints, modeltype)

    for keypoint in keypoints:
        output_img = draw_skeleton(output_img, keypoint, modeltype)

    return output_img


def draw_skeleton(img, keypoints, modeltype):
    skeleton, colors = get_vis_info(modeltype)

    scale = 1/150
    thinkness = min(int(img.shape[0]*scale), int(img.shape[1]*scale))

    for i, segment in enumerate(skeleton):
        point1_id, point2_id = segment

        point1 = keypoints[point1_id]
        point2 = keypoints[point2_id]

        color = colors[i]

        if valid_point(point1):

            cv2.circle(img, (int(point1[0]), int(point1[1])), radius=int(thinkness*1.2), color=color, thickness=-1, lineType=cv2.LINE_AA)

        if valid_point(point2):
            cv2.circle(img, (int(point2[0]), int(point2[1])), radius=int(thinkness*1.2), color=color, thickness=-1, lineType=cv2.LINE_AA)

        if not valid_point(point1) or not valid_point(point2):
            continue
        img = cv2.line(img, (int(point1[0]), int(point1[1])),
                       (int(point2[0]), int(point2[1])),
                       color, thickness=thinkness, lineType=cv2.LINE_AA)

    return img


def draw_heatmap(img, heatmap, mask_alpha=0.4):
    # Normalize the heatmap from 0 to 255
    min, max = np.min(heatmap), np.max(heatmap)
    heatmap_norm = np.uint8(255 * (heatmap - min) / (max - min))

    # Apply color to the heatmap
    color_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_MAGMA)

    # Resize to match the image shape
    color_heatmap = cv2.resize(color_heatmap, (img.shape[1], img.shape[0]))

    # Fuse both images
    if mask_alpha == 0:
        combined_img = np.hstack((img, color_heatmap))
    else:
        combined_img = cv2.addWeighted(img, mask_alpha, color_heatmap, (1 - mask_alpha), 0)

    return combined_img
