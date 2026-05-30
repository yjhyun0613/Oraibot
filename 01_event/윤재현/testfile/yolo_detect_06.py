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
from nav2_msgs.srv import ManageLifecycleNodes

class YoloNavAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_nav_alert_node')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        
        self.latest_depth = None
        self.nav_paused = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        # 서비스 및 퍼블리셔 설정 [cite: 2026-04-16]
        self.lifecycle_client = self.create_client(ManageLifecycleNodes, '/lifecycle_manager_navigation/manage_nodes')
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 토픽 구독 (OAK-D 표준 프리뷰 토픽) [cite: 2026-04-16]
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("YOLO Monitor Node Started. Distance will be displayed on boxes.")

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
        """정지 순간 알림 메시지와 이미지를 딱 한 번 전송 [cite: 2026-04-16]"""
        msg = String()
        msg.data = f"STOP: Object detected at {distance:.2f}m"
        self.alert_pub.publish(msg)
        
        try:
            if img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn(f"Alert sent! Distance: {distance:.2f}m")
        except Exception as e:
            self.get_logger().error(f"Image Publish Error: {e}")

    def control_nav(self, command):
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5):
            return
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
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    current_obj_dist = -1.0
                    if depth_img is not None:
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx, ty = int(cx * w_d / w_i), int(cy * h_d / h_i)
                        
                        dist_val = depth_img[ty, tx]
                        current_obj_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)
                        
                        if 0.1 < current_obj_dist < 10.0:
                            min_dist = min(min_dist, current_obj_dist)
                            detected_in_range = True

                    # --- 모니터링 화면 시각화 (박스 위에 거리 표시) --- [cite: 2026-04-16]
                    dist_str = f"{current_obj_dist:.2f}m" if current_obj_dist > 0 else "N/A"
                    label = f"{self.classNames[cls]} | {dist_str}"
                    
                    # 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # 텍스트 배경 (가독성용)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + 180, y1), (0, 0, 255), -1)
                    # 거리 정보 텍스트 입력
                    cv2.putText(img, label, (x1 + 5, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- 제어 로직 (3m 정지 / 5m 재개) [cite: 2026-04-16] ---
            if detected_in_range and min_dist < 3.0:
                if not self.nav_paused:
                    self.control_nav("pause")
                    self.nav_paused = True
                    self.send_alert_once(img, min_dist)
            elif not detected_in_range or min_dist >= 5.0:
                if self.nav_paused:
                    self.control_nav("resume")
                    self.nav_paused = False

            cv2.imshow("AMR Object Detection Monitor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

def main():
    rclpy.init()
    model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    
    node = YoloNavAlertNode(model)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()