import os
import sys
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloNavAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_nav_alert_node')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        
        self.latest_depth = None
        self.nav_paused = False

        # 서비스 및 퍼블리셔 설정
        from nav2_msgs.srv import ManageLifecycleNodes
        self.lifecycle_client = self.create_client(ManageLifecycleNodes, '/lifecycle_manager_navigation/manage_nodes')
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 토픽 구독
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def send_alert_once(self, img, distance):
        """정지 시점의 스냅샷을 복사하여 즉시 전송 [cite: 2026-04-16]"""
        msg = String()
        msg.data = f"STOP: Object at {distance:.2f}m"
        self.alert_pub.publish(msg)
        
        try:
            if img is not None:
                # 검은 화면 방지를 위해 이미지 데이터 복사 및 유효성 검사 [cite: 2026-04-16]
                alert_frame = img.copy() 
                img_msg = self.bridge.cv2_to_imgmsg(alert_frame, encoding="bgr8")
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn("Alert Image Sent Successfully.")
        except Exception as e:
            self.get_logger().error(f"Alert Image Fail: {e}")

    def control_nav(self, command):
        from nav2_msgs.srv import ManageLifecycleNodes
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5): return
        req = ManageLifecycleNodes.Request()
        req.command = ManageLifecycleNodes.Request.PAUSE if command == "pause" else ManageLifecycleNodes.Request.RESUME
        self.lifecycle_client.call_async(req)

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
                        dist_val = depth_img[ty, tx]
                        current_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)
                        
                        if 0.1 < current_dist < 10.0:
                            min_dist = min(min_dist, current_dist)
                            detected_in_range = True

                    # --- 시각화 수정: 빨간 박스 배경 제거, 숫자만 표시 [cite: 2026-04-16] ---
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if current_dist > 0:
                        dist_text = f"{current_dist:.2f}m"
                        # 텍스트 그림자 효과 (검은색 외곽선 역할)를 주어 가독성 확보
                        cv2.putText(img, dist_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3) # 두꺼운 검정색
                        cv2.putText(img, dist_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # 얇은 녹색

            # 제어 로직
            if detected_in_range and min_dist < 3.0:
                if not self.nav_paused:
                    self.control_nav("pause")
                    self.nav_paused = True
                    # 멈춘 시점의 프레임을 별도로 캡처하여 전송 [cite: 2026-04-16]
                    self.send_alert_once(img, min_dist)
            elif not detected_in_range or min_dist >= 5.0:
                if self.nav_paused:
                    self.control_nav("resume")
                    self.nav_paused = False

            cv2.imshow("Monitor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

def main():
    rclpy.init()
    model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    node = YoloNavAlertNode(model)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()