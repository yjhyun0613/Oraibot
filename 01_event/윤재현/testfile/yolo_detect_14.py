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
        
        # 스레드 간 데이터 충돌 방지를 위한 Lock 객체 생성
        self.lock = threading.Lock()
        self.latest_depth = None
        
        self.last_status = "CLEAR"
        self.last_detect_time = 0.0
        self.clear_patience = 1.0  

        # --- 콜백 그룹 설정 (각 콜백이 서로를 블로킹하지 않도록 독립적인 스레드에 할당) ---
        self.rgb_cb_group = MutuallyExclusiveCallbackGroup()
        self.depth_cb_group = MutuallyExclusiveCallbackGroup()
        self.pub_cb_group = MutuallyExclusiveCallbackGroup()

        self.important_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publisher
        self.status_pub = self.create_publisher(
            String, '/yolo/detection_status', self.important_qos, callback_group=self.pub_cb_group)
        self.alert_img_pub = self.create_publisher(
            Image, '/yolo/alert_image', self.important_qos, callback_group=self.pub_cb_group)

        # Subscriber (각각 다른 콜백 그룹 지정)
        self.create_subscription(
            CompressedImage, '/robot4/oakd/rgb/image_raw/compressed', 
            self.rgb_callback, 10, callback_group=self.rgb_cb_group)
        self.create_subscription(
            Image, '/robot4/oakd/stereo/image_raw', 
            self.depth_callback, 10, callback_group=self.depth_cb_group)

        # YOLO 탐지 루프 스레드 실행
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
            # depth 데이터는 다른 스레드(detection_loop)에서도 읽으므로 Lock 처리
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

            # 최신 depth 이미지를 Lock을 걸고 안전하게 가져옴
            with self.lock:
                depth_img = self.latest_depth.copy() if self.latest_depth is not None else None

            # YOLO 추론
            results = self.model.predict(img, stream=True, verbose=False)
            min_dist = float('inf')
            detected_in_range = False

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

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if current_dist > 0:
                        cv2.putText(img, f"{current_dist:.2f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
    model_path = '/home/rokey/yjh/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    
    node = YoloDetectionAlertNode(model)
    
    # --- MultiThreadedExecutor 적용 ---
    # 병렬로 처리할 스레드 개수를 지정합니다. (기본값은 시스템 코어 수)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        # 기존의 rclpy.spin_once() 루프 대신 executor.spin() 사용
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True # Python 쓰레드 종료 플래그
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()