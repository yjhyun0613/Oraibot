import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class SafeStop(Node):
    def __init__(self):
        super().__init__('safe_stop')

        self.sub = self.create_subscription(
            LaserScan, 'robot5/scan', self.scan_cb, 10)

        self.pub = self.create_publisher(
            Twist, 'robot5/cmd_vel', 10)

        # 파라미터
        self.STOP_DIST = 0.5
        self.RELEASE_DIST = 0.7
        self.FRAME_THRESHOLD = 3

        self.stop_counter = 0
        self.release_counter = 0
        self.is_stopped = False

    def scan_cb(self, msg):
        # 1️⃣ 노이즈 제거
        valid = [r for r in msg.ranges if 0.05 < r < 10.0]
        if len(valid) < 5:
            return

        # 2️⃣ 중앙값 필터
        dist = np.median(valid)

        # 3️⃣ 상태 머신 (히스테리시스 + debounce)
        if dist < self.STOP_DIST:
            self.stop_counter += 1
            self.release_counter = 0
        elif dist > self.RELEASE_DIST:
            self.release_counter += 1
            self.stop_counter = 0
        else:
            self.stop_counter = 0
            self.release_counter = 0

        # 4️⃣ 정지 조건
        if self.stop_counter >= self.FRAME_THRESHOLD:
            self.is_stopped = True

        # 5️⃣ 해제 조건
        if self.release_counter >= self.FRAME_THRESHOLD:
            self.is_stopped = False

        # 6️⃣ 출력
        cmd = Twist()

        if self.is_stopped:
            self.get_logger().warn(f"STOP! dist={dist:.2f}")
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.2  # 기본 속도

        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = SafeStop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()