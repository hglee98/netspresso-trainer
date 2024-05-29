import cv2
import numpy as np


def _voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class DetectionVisualizer:
    def __init__(self, class_map):
        self.cmap = _voc_color_map(256)
        self.class_map = class_map

    def draw(self, image, detections):
        bbox_label, class_label = detections[0], detections[1]

        # Get bbox coordinates
        x1 = int(bbox_label[0])
        y1 = int(bbox_label[1])
        x2 = int(bbox_label[2])
        y2 = int(bbox_label[3])

        # Get bbox color
        color = self.cmap[class_label].tolist()

        # Draw bbox
        visualize_image = cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)

        # Get class name
        class_name = self.class_map[class_label] if self.class_map else str(class_label)

        # Draw class info
        text_size, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        visualize_image = cv2.rectangle(visualize_image, (x1, y1-5-text_h), (x1+text_w, y1), color=color, thickness=-1)
        visualize_image = cv2.putText(visualize_image, str(class_name), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return visualize_image

    def visualize(self, image):
        cv2.imshow('Detecttion result', image)
        cv2.waitKey(1)