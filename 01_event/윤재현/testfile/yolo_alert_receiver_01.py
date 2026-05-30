import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class YoloReceiver(Node):
    def __init__(self):
        super().__init__('yolo_receiver_node')
        self.bridge = CvBridge()
        self.latest_alert_img = None
        
        self.create_subscription(String, '/yolo/alert_status', self.txt_cb, 10)
        self.create_subscription(Image, '/yolo/alert_image', self.img_cb, 10)
        
        # 화면 갱신을 위한 타이머 (10Hz)
        self.timer = self.create_timer(0.1, self.draw_callback)
        self.get_logger().info("Receiver is waiting for a stop event...")

    def txt_cb(self, msg):
        self.get_logger().warn(f"EVENT RECEIVED: {msg.data}")

    def img_cb(self, msg):
        try:
            # 수신 이미지를 bgr8로 복사하여 저장
            new_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if new_img is not None:
                self.latest_alert_img = np.array(new_img, copy=True)
                self.get_logger().info("New alert photo captured!")
        except Exception as e:
            self.get_logger().error(f"Receive Error: {e}")

    def draw_callback(self):
        """저장된 이미지가 있으면 계속 화면에 출력"""
        if self.latest_alert_img is not None:
            cv2.imshow("LAST_STOP_FRAME", self.latest_alert_img)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloReceiver()
    try: rclpy.spin(node)
    except: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()