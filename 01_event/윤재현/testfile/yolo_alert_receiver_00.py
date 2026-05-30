import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class YoloReceiver(Node):
    def __init__(self):
        super().__init__('yolo_receiver_node')
        self.bridge = CvBridge()
        self.create_subscription(String, '/yolo/alert_status', self.txt_cb, 10)
        self.create_subscription(Image, '/yolo/alert_image', self.img_cb, 10)
        self.get_logger().info("Receiver ready. Waiting for STOP signal...")

    def txt_cb(self, msg):
        self.get_logger().warn(f"ALERT RECEIVED: {msg.data}")

    def img_cb(self, msg):
        try:
            # ROS 이미지를 OpenCV 이미지로 변환 [cite: 2026-01-01]
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img is not None:
                cv2.imshow("Captured STOP Frame", cv_img)
                # 창이 꺼지지 않도록 충분히 대기하거나 수동으로 끌 때까지 유지 [cite: 2026-01-01]
                cv2.waitKey(1) 
        except Exception as e:
            self.get_logger().error(f"Image View Error: {e}")

def main():
    rclpy.init()
    node = YoloReceiver()
    try: rclpy.spin(node)
    except: pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__': main()