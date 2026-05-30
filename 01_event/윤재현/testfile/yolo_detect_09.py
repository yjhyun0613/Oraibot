import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import os
from queue import Queue, Empty
import threading
from nav2_msgs.srv import ManageLifecycleNodes

class YoloSender(Node):
    def __init__(self):
        super().__init__('yolo_sender_node')
        # 한양대 프로젝트 경로 반영
        model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
        self.model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
        
        self.bridge = CvBridge()
        self.img_q = Queue(maxsize=1)
        self.latest_depth = None
        self.nav_paused = False # 딱 한 장만 보내기 위한 상태 플래그

        self.lifecycle_client = self.create_client(ManageLifecycleNodes, '/lifecycle_manager_navigation/manage_nodes')
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 토픽 구독
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_cb, 10)
        self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_cb, 10)

        threading.Thread(target=self.det_loop, daemon=True).start()

    def rgb_cb(self, msg):
        try:
            # 수신 시 bgr8로 확실히 변환
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.img_q.full(): 
                self.img_q.put(cv_img)
        except: pass

    def depth_cb(self, msg):
        try: 
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except: pass

    def send_alert_photo(self, img, dist):
        """검은 화면 방지를 위한 강제 데이터 고정 및 전송"""
        try:
            # 1. 텍스트 알림
            t_msg = String()
            t_msg.data = f"STOP_EVENT: {dist:.2f}m"
            self.alert_pub.publish(t_msg)

            # 2. 이미지 딥카피 및 타입 강제 지정 (검은 화면 방지 핵심)
            if img is not None:
                # 메모리 참조를 완전히 끊고 새 객체 생성
                alert_frame = np.array(img, copy=True, dtype=np.uint8) 
                
                # ROS 이미지 메시지로 변환
                img_msg = self.bridge.cv2_to_imgmsg(alert_frame, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "camera_frame"
                
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn(f"!!! ALERT PHOTO SENT ({dist:.2f}m) !!!")
        except Exception as e:
            self.get_logger().error(f"Photo Send Failed: {e}")

    def det_loop(self):
        while rclpy.ok():
            try:
                img = self.img_q.get(timeout=0.5)
            except Empty: continue

            results = self.model.predict(img, stream=True, verbose=False)
            min_d = float('inf')
            found = False

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    dist = -1.0
                    if self.latest_depth is not None:
                        h_d, w_d = self.latest_depth.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx, ty = int(cx * w_d / w_i), int(cy * h_d / h_i)
                        try:
                            d_val = self.latest_depth[ty, tx]
                            dist = d_val / 1000.0 if d_val > 100 else float(d_val)
                            if 0.1 < dist < 10.0:
                                min_d = min(min_d, dist)
                                found = True
                        except: pass

                    # 화면 표시: 숫자만 깔끔하게
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                    if dist > 0:
                        cv2.putText(img, f"{dist:.2f}m", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # 딱 한 번만 실행되는 로직
            if found and min_d < 3.0:
                if not self.nav_paused:
                    self.send_alert_photo(img, min_d) # 이미지 먼저 전송
                    self.nav_paused = True
                    self.call_nav("pause")
            elif not found or min_d >= 5.0:
                if self.nav_paused:
                    self.nav_paused = False
                    self.call_nav("resume")

            cv2.imshow("AMR_Sender_Monitor", img)
            cv2.waitKey(1)

    def call_nav(self, cmd):
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5): return
        req = ManageLifecycleNodes.Request()
        req.command = 2 if cmd == "pause" else 3
        self.lifecycle_client.call_async(req)

def main():
    rclpy.init()
    node = YoloSender()
    try: rclpy.spin(node)
    except: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()