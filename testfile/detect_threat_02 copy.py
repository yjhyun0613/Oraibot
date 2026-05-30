import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class Robot5SafetyNode(Node):
    def __init__(self):
        super().__init__('robot5_safety_node')
        
        # 1. Nav2의 속도 명령을 구독 (Nav2 설정에서 출력 토픽을 이 이름으로 바꿔야 함)
        # 만약 바꾸지 못한다면 이 노드는 단순히 '감시'만 하고 멈춤 명령만 쏠 수 있습니다.
        self.nav_sub = self.create_subscription(
            Twist, 
            '/robot5/cmd_vel_nav', 
            self.nav_callback, 
            10)
        
        # 2. LiDAR 데이터 구독 (/robot5/scan)
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/robot5/scan', 
            self.scan_callback, 
            10)
        
        # 3. 실제 로봇에게 전달할 최종 속도 발행 (/robot5/cmd_vel)
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        
        self.is_obstacle_detected = False
        self.stop_distance = 0.5  # 50cm
        
        # 감시 범위: 0도(정면) ~ 90도(왼쪽)
        self.min_angle_rad = math.radians(0.0)
        self.max_angle_rad = math.radians(90.0)

    def scan_callback(self, msg):
        # 인덱스 계산
        lower_idx = int((self.min_angle_rad - msg.angle_min) / msg.angle_increment)
        upper_idx = int((self.max_angle_rad - msg.angle_min) / msg.angle_increment)
        
        lower_idx = max(0, lower_idx)
        upper_idx = min(len(msg.ranges), upper_idx)
        
        # 0도 ~ 90도 범위 추출
        relevant_scan = msg.ranges[lower_idx:upper_idx]
        
        # 유효 거리 필터링 (0.0 이나 inf 제외)
        valid_ranges = [r for r in relevant_scan if msg.range_min < r < msg.range_max]
        
        if valid_ranges and min(valid_ranges) < self.stop_distance:
            if not self.is_obstacle_detected:
                self.get_logger().warn(f"[/robot5] 왼쪽 전방 장애물 감지! 거리: {min(valid_ranges):.2f}m")
            self.is_obstacle_detected = True
        else:
            self.is_obstacle_detected = False

    def nav_callback(self, nav_msg):
        # Nav2에서 온 명령을 가로채서 판단
        final_msg = Twist()
        
        if self.is_obstacle_detected:
            # 장애물 있으면 강제 정지
            final_msg.linear.x = 0.0
            final_msg.angular.z = 0.0
        else:
            # 안전하면 Nav2 명령 그대로 전달
            final_msg = nav_msg
            
        self.cmd_pub.publish(final_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Robot5SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 안전을 위해 멈춤 명령 발행
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()