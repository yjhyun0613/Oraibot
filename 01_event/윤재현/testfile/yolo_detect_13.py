import os
import time  # 시간 체크를 위해 추가
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
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
        self.latest_depth = None
        
        # 이전 상태를 저장하여 메시지 중복 발행 방지
        self.last_status = "CLEAR"
        
        # 깜빡임(Flickering) 방지를 위한 타이머 변수 추가
        self.last_detect_time = 0.0
        self.clear_patience = 2.0  # 사람이 사라지거나 멀어진 후 CLEAR로 인정하기까지 기다릴 시간 (초 단위)

        # 상태 알림 토픽 (String)
        self.status_pub = self.create_publisher(String, '/yolo/detection_status', 10)
        # 시각화된 이미지 토픽
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 토픽 구독
        self.create_subscription(CompressedImage, '/robot4/oakd/rgb/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/robot4/oakd/stereo/image_raw', self.depth_callback, 10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Detection Node Started. Monitoring only.")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is not None:
                if self.image_queue.full():
                    try: self.image_queue.get_nowait()
                    except: pass
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Callback Error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def publish_status(self, status_text):
        """상태가 변경될 때만 혹은 주기적으로 토픽 발행"""
        msg = String()
        msg.data = status_text
        self.status_pub.publish(msg)

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            results = self.model.predict(img, stream=True, verbose=False)
            min_dist = float('inf')
            detected_in_range = False
            depth_img = self.latest_depth

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
                            # mm 단위를 m 단위로 변환
                            current_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)
                            if 0.1 < current_dist < 10.0:
                                min_dist = min(min_dist, current_dist)
                                detected_in_range = True
                        except: pass

                    # 시각화 (선택 사항)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if current_dist > 0:
                        cv2.putText(img, f"{current_dist:.2f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- 핵심 로직: 타이머를 적용한 디바운싱 ---
            current_time = time.time()
            is_detected = detected_in_range and min_dist < 3.0
            
            if is_detected:
                # 3미터 내에 탐지되면 마지막 탐지 시간을 현재 시간으로 갱신
                self.last_detect_time = current_time
                
                # 상태가 CLEAR였다가 처음 DETECTED가 된 순간에만 발행 (딱 한 번)
                if self.last_status != "DETECTED":
                    self.get_logger().warn(f"EVENT: Object detected! Dist: {min_dist:.2f}m")
                    
                    # 1) 텍스트 메시지 전송
                    self.publish_status(f"DETECTED: {min_dist:.2f}m")
                    
                    # 2) 알림용 이미지 전송
                    try:
                        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                        self.alert_img_pub.publish(img_msg)
                    except Exception as e:
                        self.get_logger().error(f"Image Publish Error: {e}")
                    
                    self.last_status = "DETECTED"
            else:
                # 탐지가 안 되거나 3미터 밖으로 나갔을 때
                # 무조건 바로 CLEAR로 바꾸지 않고, 마지막 탐지 이후 지정된 시간(2초)이 지났는지 확인
                if self.last_status == "DETECTED":
                    time_since_last_detect = current_time - self.last_detect_time
                    if time_since_last_detect > self.clear_patience:
                        self.get_logger().info("EVENT: Path is clear. Object moved away.")
                        
                        # 1) 클리어 메시지 딱 한 번 전송
                        self.publish_status("CLEAR")
                        
                        # 상태를 CLEAR로 변경 (이제 다시 탐지되면 이벤트가 발생함)
                        self.last_status = "CLEAR"

            # 실시간 모니터링
            cv2.imshow("Detection Monitor", img)
            cv2.waitKey(1) # imshow 업데이트를 위해 필요 (스레드 환경이므로 짧게 유지)

def main():
    rclpy.init()
    # 경로 수정 필요
    model_path = '/home/rokey/yjh/yolo8n_amr_huma1n.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    node = YoloDetectionAlertNode(model)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()