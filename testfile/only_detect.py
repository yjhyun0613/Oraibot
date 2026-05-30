import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class PureLidarTestNode(Node):
    def __init__(self):
        super().__init__('pure_lidar_test_node')
        
        # --- 최소/최대 거리 설정 ---
        self.min_distance = 0.20 # 20cm (로봇 자기 몸통은 무시!)
        self.max_distance = 0.60 # 60cm (여기까지만 감지)
        
        self.min_angle_deg = -180.0 # 정면 기준 우측 40도
        self.max_angle_deg = 0.0  # 정면 기준 좌측 40도
        
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/robot5/scan', 
            self.scan_callback, 
            10
        )
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🔍 순수 라이다 감지 테스트 (자기 몸통 무시 버전)")
        self.get_logger().info(f" - 감지 거리: {self.min_distance}m ~ {self.max_distance}m")
        self.get_logger().info(f" - 감지 각도: {self.min_angle_deg}도 ~ {self.max_angle_deg}도")
        self.get_logger().info("=========================================")

    def scan_callback(self, msg):
        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            # 🚨 0.2m 보다 크고 0.6m 보다 작은 것만 찾습니다.
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                # 설정한 각도 범위 안에 들어오는지 확인
                if min_rad <= angle <= max_rad:
                    self.get_logger().info(f"🎯 삐빅! 실제 장애물 감지됨 -> 거리: {dist:.2f}m, 각도: {math.degrees(angle):.1f}도")
                    break 

def main(args=None):
    rclpy.init(args=args)
    node = PureLidarTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()