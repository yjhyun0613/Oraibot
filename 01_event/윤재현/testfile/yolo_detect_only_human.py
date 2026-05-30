import os
import sys
import time # 시간 측정을 위해 추가
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
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
        self.last_person_time = 0.0 # 사람을 마지막으로 감지한 시간을 기록할 변수 추가

        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 조이스틱 우선순위 토픽 (터틀봇4)
        self.cmd_pub = self.create_publisher(Twist, '/robot3/teleop_cmd_vel', 10)
        
        # 0.1초마다 멈춤 명령 연사 (nav_paused가 True일 때만)
        self.timer = self.create_timer(0.1, self.stop_timer_callback)

        self.create_subscription(CompressedImage, '/robot3/oakd/rgb/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/robot3/oakd/stereo/image_raw', self.depth_callback, 10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Node initialized. Monitoring with Cooldown Logic.")

    def stop_timer_callback(self):
        if self.nav_paused:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_pub.publish(stop_msg)

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
            self.get_logger().error(f"Depth Callback Error: {e}")

    def send_alert_once(self, img, distance):
        msg = String()
        msg.data = f"STOP: Person detected at {distance:.2f}m"
        self.alert_pub.publish(msg)
        try:
            if img is not None and img.size > 0:
                alert_frame = np.array(img, copy=True)
                if alert_frame.dtype != np.uint8:
                    alert_frame = alert_frame.astype(np.uint8)
                img_msg = self.bridge.cv2_to_imgmsg(alert_frame, encoding="bgr8")
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn(f"Alert image published! Distance: {distance:.2f}m")
        except Exception as e:
            self.get_logger().error(f"Failed to publish alert image: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            results = self.model.predict(img, classes=[0], stream=True, verbose=False)
            
            min_dist = float('inf')
            detected_person_in_range = False
            depth_img = self.latest_depth

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None: continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    
                    if self.model.names[cls_id] != 'person':
                        continue

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
                                detected_person_in_range = True
                        except: pass

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if current_dist > 0:
                        dist_text = f"PERSON | {current_dist:.2f}m"
                        cv2.putText(img, dist_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(img, dist_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- 제어 로직 (Cooldown 적용) ---
            current_time = time.time()

            if detected_person_in_range and min_dist < 1.5:
                self.last_person_time = current_time # 사람을 감지할 때마다 현재 시간을 갱신
                
                if not self.nav_paused:
                    self.get_logger().info(f"STOP: Person detected at {min_dist:.2f}m. Triggering override.")
                    self.send_alert_once(img, min_dist)
                    self.nav_paused = True
                    
            elif self.nav_paused:
                # 정지 상태인데 카메라에서 사람이 안 보이거나 멀어진 경우
                # 마지막으로 사람을 본 지 1.5초가 지났는지 확인
                if (current_time - self.last_person_time) > 1.5:
                    self.get_logger().info("RESUME: 1.5s clear. Releasing override.")
                    self.nav_paused = False

            cv2.imshow("Person Detection Monitor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break

def main():
    rclpy.init()
    model_path = '/home/rokey/rokey_ws/src/main_proj/yolo/yolo8n_amr_human1.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    
    node = YoloNavAlertNode(model)
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