import cv2
from yolo_model_handler import YOLOModelHandler

class VideoProcessor:
    def __init__(self, video_path, config_path, weights_path, class_names):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.model_handler = YOLOModelHandler(config_path, weights_path, class_names)

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.model_handler.detect_objects(frame)
            for detection in detections:
                x, y, w, h = detection['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model_handler.class_names[detection['class_id']]}: {detection['confidence']:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Fish Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    video_processor = VideoProcessor('path_to_video.mp4', 'path_to_yolo_config.cfg', 'path_to_yolo_weights.weights', ['fish'])
    video_processor.process_video()
