import cv2
import numpy as np

class YOLOModelHandler:
    def __init__(self, config_path, weights_path, class_names):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.class_names = class_names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image):
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, width, height = list(map(int, detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])))
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        results = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            results.append({'class_id': class_ids[i], 'confidence': confidences[i], 'box': box})

        return results
