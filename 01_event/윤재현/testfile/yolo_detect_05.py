import os
import sys
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path
import cv2
from nav2_msgs.srv import ManageLifecycleNodes

class YoloNavAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_nav_alert_node')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']
        
        self.latest_depth = None
        self.nav_paused = False

        # 로봇 설정 및 서비스 클라이언트
        ns = '/robot3'
        self.lifecycle_client = self.create_client(
            ManageLifecycleNodes, 
            '/lifecycle_manager_navigation/manage_nodes'
        )

        # 알림 퍼블리셔 추가 [cite: 2026-04-16]
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 데이터 구독 (사용자 환경에 맞춰 토픽명 수정 가능)
        self.rgb_subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.rgb_callback,
            10)

        self.depth_subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/depth',
            self.depth_callback,
            10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is not None and img.size > 0:
                if self.image_queue.full():
                    self.image_queue.get()
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Callback Error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth Callback Error: {e}")

    def control_nav(self, command):
        """Nav2 Lifecycle Manager를 통한 정지/재개 서비스 호출 [cite: 2026-04-16]"""
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5):
            return

        req = ManageLifecycleNodes.Request()
        if command == "pause":
            req.command = ManageLifecycleNodes.Request.PAUSE
            self.nav_paused = True
        else:
            req.command = ManageLifecycleNodes.Request.RESUME
            self.nav_paused = False
            
        self.lifecycle_client.call_async(req)
        self.get_logger().info(f"Nav2 {command.upper()} sent.")

    def send_alert(self, img, distance):
        """객체 감지 시 알림 메시지와 이미지를 전송 [cite: 2026-04-16]"""
        # 텍스트 알림
        text_msg = String()
        text_msg.data = f"[ALERT] Object detected at {distance:.2f}m. Robot Stopped."
        self.alert_pub.publish(text_msg)

        # 이미지 전송
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.alert_img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Alert image publish fail: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            results = self.model.predict(img, stream=True, verbose=False)
            
            min_dist = float('inf')
            detected = False
            depth_img = self.latest_depth

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None: continue
                
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if depth_img is not None:
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx, ty = int(cx * w_d / w_i), int(cy * h_d / h_i)

                        if 0 <= tx < w_d and 0 <= ty < h_d:
                            pixel_val = depth_img[ty, tx]
                            dist_m = pixel_val / 1000.0 if pixel_val > 100 else float(pixel_val)
                            
                            if dist_m > 0.1:
                                min_dist = min(min_dist, dist_m)
                                detected = True

                    # 시각화 (원본 이미지에 박스 그림)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # --- 거리 기반 제어 및 알림 로직 [cite: 2026-04-16] ---
            if detected:
                if min_dist < 3.0: 
                    if not self.nav_paused:
                        self.control_nav("pause")
                        self.send_alert(img, min_dist) # 멈출 때 이미지 전송
                elif min_dist >= 5.0:
                    if self.nav_paused:
                        self.control_nav("resume")
            else:
                if self.nav_paused:
                    self.control_nav("resume")

            if img.size > 0:
                cv2.imshow("Robot Monitor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break

def main():
    model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')

    rclpy.init()
    node = YoloNavAlertNode(model)
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        node.thread.join()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()