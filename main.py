from typing import List, Tuple, Any, Dict
import cv2
from cv2 import VideoCapture
from numpy import ndarray
from ultralytics import YOLO, solutions
import numpy as np
import os
from MockVideoWriter import MockVideoWriter

class Analyse:
    """
    Analyse: Class for real-time vehicle detection, counting, and analysis using YOLOv8. model
    and visualizing the detection with analytics.
    """

    def __init__(self) -> None:
        """
        Initialize the Analyse class with default values for
        class-wise count and video writer.
        """
        self.clswise_count = {}
        self.analysis_frame = MockVideoWriter()
        pass

    def Yolo_model(self, model_path: os.path = "/Users/mac/PycharmProjects/computer vision project/video_analysis/yolov8s.pt") -> YOLO:
        """
        Load the YOLOv8 model.

        Args:
            model_path: Path to the YOLOv8 weights file.

        Returns:
            YOLO: Loaded YOLO model.
        """
        self.model = YOLO(model_path)
        return self.model

    def setup_frame_and_ultralytics(self, video_path: os.path) -> Tuple[VideoCapture, Any, Any, Any]:
        """
        Set up the video capture and get video properties.

        Args:
            video_path: Path to the video file.

        Returns:
            tuple: Video capture object, width, height, and FPS of the video.
        """
        self.cap = cv2.VideoCapture(video_path)
        assert self.cap.isOpened(), "Error reading video file"
        w, h, fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        return self.cap, w, h, fps

    def yolo_analytics(self, clswise_count: Dict, w: int, h: int) -> None:
        """
        Update analytics with the current class-wise count.

        Args:
            clswise_count: Dictionary with count of detected classes.
            w: Width of the image/frame.
            h: Height of the image/frame.
        """
        analytics = solutions.Analytics(
            type="bar",
            writer=self.analysis_frame,
            im0_shape=(w, h),
            view_img=False,
            bg_color="black",
            fg_color="white",
        )
        analytics.update_bar(clswise_count)

    def process_frame(self, frame: ndarray, model: YOLO, w: int, h: int) -> ndarray:
        """
        Process a single video frame for object bluring.

        Args:
            frame: Input video frame.
            model: YOLO model for detection.
            w: Width of the video frame.
            h: Height of the video frame.

        Returns:
            ndarray: Annotated video frame.
        """
        results = self.model.track(frame, persist=True, verbose=True)
        if results[0].boxes:  # Check if there are any boxes detected
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()

            for box, cls, score in zip(boxes, clss, scores):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                detected_object = model.names[int(cls)]

                if detected_object == "car" or detected_object == "bus" or detected_object=="truck":
                    label = f"{detected_object} {score:.2f} "
                    # Update class-wise count
                    self.clswise_count[model.names[int(cls)]] = self.clswise_count.get(model.names[int(cls)], 0) + 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # y1 = max(y1, label_size[1] + 100)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1 + base_line - 10), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                else:
                    pass
            self.yolo_analytics(self.clswise_count, w, h)
            self.clswise_count = {}

        return frame

    def combined_result_frame(self, main_image_frame: ndarray, analytics_frame: ndarray) -> ndarray:
        """
        Combine the main detection frame with the analytics frame.

        Args:
            main_image_frame: Frame with detected objects.
            analytics_frame: Frame showing analytics.

        Returns:
            ndarray: Combined frame with both detection and analytics.
        """
        new_frame_height, new_frame_width = 870, 1440
        analytics_frame_height, analytics_frame_width = 350, 460

        resized_main_frame = cv2.resize(analytics_frame, (analytics_frame_width, analytics_frame_height))
        resized_analytics_frame = cv2.resize(main_image_frame, (new_frame_width, new_frame_height))

        combined_frame = np.zeros((new_frame_height, new_frame_width, 3), dtype=np.uint8)
        combined_frame[:new_frame_height, :new_frame_width] = resized_analytics_frame
        x_offset = new_frame_width - analytics_frame_width  # Place resized_frame so it touches the right edge
        y_offset = 0  # Top edge
        combined_frame[y_offset:y_offset + analytics_frame_height, x_offset:x_offset + analytics_frame_width] = resized_main_frame

        return combined_frame

    def run(self, video_path: os.path) -> None:
        """
        Run the YOLO-based analytics on the provided video.

        Args:
            video_path: Path to the video file to be processed.

        """
        Yolo_model = self.Yolo_model()
        cap, w, h, fps = self.setup_frame_and_ultralytics(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                detection_frame = self.process_frame(frame, Yolo_model, w, h)
                analysis_frame = self.analysis_frame.frame

                combined_frame = self.combined_result_frame(main_image_frame=detection_frame, analytics_frame=analysis_frame)

                cv2.imshow("frame", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    file = "vid.mp4"
    model = Analyse()
    model.run(video_path=file)


