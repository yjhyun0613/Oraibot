import os
import time
import rclpy
import threading
from queue import Queue, Empty
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class YoloDetectionAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_detection_alert_node')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        
        self.lock = threading.Lock()
        self.latest_depth = None
        
        self.last_status = "CLEAR"
        self.last_detect_time = 0.0
        self.clear_patience = 1.0  

        self.rgb_cb_group = MutuallyExclusiveCallbackGroup()
        self.depth_cb_group = MutuallyExclusiveCallbackGroup()
        self.pub_cb_group = MutuallyExclusiveCallbackGroup()

        self.important_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.status_pub = self.create_publisher(
            String, '/yolo/detection_status', self.important_qos, callback_group=self.pub_cb_group)
        self.alert_img_pub = self.create_publisher(
            Image, '/yolo/alert_image', self.important_qos, callback_group=self.pub_cb_group)

        self.create_subscription(
            CompressedImage, '/robot4/oakd/rgb/image_raw/compressed', 
            self.rgb_callback, 10, callback_group=self.rgb_cb_group)
        self.create_subscription(
            Image, '/robot4/oakd/stereo/image_raw', 
            self.depth_callback, 10, callback_group=self.depth_cb_group)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Detection Node Started with Multi-Threading. Monitoring only.")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is not None:
                if self.image_queue.full():
                    try: self.image_queue.get_nowait()
                    except Empty: pass
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Callback Error: {e}")

    def depth_callback(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.latest_depth = depth_img
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def publish_status(self, status_text):
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            with self.lock:
                depth_img = self.latest_depth.copy() if self.latest_depth is not None else None

            results = self.model.predict(img, stream=True, verbose=False, classes=[0])
            
            min_dist = float('inf')
            detected_in_range = False
            
            # =====================================================================
            # 변경 포인트: 한 프레임 내의 모든 탐지 결과를 먼저 저장할 리스트 생성
            # =====================================================================
            frame_detections = []

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None: continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    current_dist = -1.0
                    if depth_img is not None:
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx, ty = int(cx * w_d / w_i), int(cy * h_d / h_i)
                        
                        try:
                            dist_val = depth_img[ty, tx]
                            current_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)
                            if 0.1 < current_dist < 10.0:
                                min_dist = min(min_dist, current_dist)
                                detected_in_range = True
                        except: pass

                    # 바로 그리지 않고 리스트에 데이터(좌표와 거리) 저장
                    frame_detections.append((x1, y1, x2, y2, current_dist))

            # =====================================================================
            # 수집된 정보를 바탕으로 박스 그리기 (제일 가까우면 빨간색, 나머진 초록색)
            # =====================================================================
            for det in frame_detections:
                x1, y1, x2, y2, dist = det
                
                # 유효한 거리가 있고, 그 거리가 현재 프레임의 최소 거리(min_dist)와 일치하면 빨간색
                if dist > 0 and dist == min_dist and min_dist < float('inf'):
                    color = (0, 0, 255)  # 빨간색 (BGR)
                else:
                    color = (0, 255, 0)  # 초록색 (BGR)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if dist > 0:
                    cv2.putText(img, f"{dist:.2f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 타이머 디바운싱 및 토픽 발행 로직 ---
            current_time = time.time()
            is_detected = detected_in_range and min_dist < 3.0
            
            if is_detected:
                self.last_detect_time = current_time
                if self.last_status != "DETECTED":
                    self.get_logger().warn(f"EVENT: Object detected! Dist: {min_dist:.2f}m")
                    self.publish_status("DETECTED")
                    try:
                        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                        self.alert_img_pub.publish(img_msg)
                    except Exception as e:
                        self.get_logger().error(f"Image Publish Error: {e}")
                    
                    self.last_status = "DETECTED"
            else:
                if self.last_status == "DETECTED":
                    if (current_time - self.last_detect_time) > self.clear_patience:
                        self.get_logger().info("EVENT: Path is clear. Object moved away.")
                        self.publish_status("CLEAR")
                        self.last_status = "CLEAR"

            cv2.imshow("Detection Monitor", img)
            cv2.waitKey(1)

def main():
    rclpy.init()
    
    model = YOLO('yolov8n.pt') 
    
    node = YoloDetectionAlertNode(model)
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()