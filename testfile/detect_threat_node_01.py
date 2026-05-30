import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String  # 통신용 메시지 추가
import math

class Robot5SafetyNode(Node):
    def __init__(self):
        super().__init__('robot5_safety_node')
        
        # 주행 속도 관련 Pub/Sub
        self.nav_sub = self.create_subscription(Twist, '/robot5/cmd_vel_nav', self.nav_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        
        # --- 새로 추가된 GUI 팀과의 통신 ---
        # 1. GUI 팀에게 경고 보내기
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        # 2. GUI 팀으로부터 재개 명령 받기
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        # 상태 변수
        self.is_waiting_for_gui = False  # True면 GUI 허락을 기다리는 중 (정지 상태 유지)
        self.stop_distance = 0.6  # 60cm
        
        # 감시 범위 우상단: -90도(오른쪽) ~ 180도(후방) 
        self.min_angle_rad = math.radians(-180.0)
        self.max_angle_rad = math.radians(0.0)

    def scan_callback(self, msg):
        # 대기 상태일 때는 라이다 판단을 멈추고 계속 대기
        if self.is_waiting_for_gui:
            return

        lower_idx = int((self.min_angle_rad - msg.angle_min) / msg.angle_increment)
        upper_idx = int((self.max_angle_rad - msg.angle_min) / msg.angle_increment)
        lower_idx = max(0, lower_idx)
        upper_idx = min(len(msg.ranges), upper_idx)
        
        relevant_scan = msg.ranges[lower_idx:upper_idx]
        valid_ranges = [r for r in relevant_scan if msg.range_min < r < msg.range_max]
        
        if valid_ranges and min(valid_ranges) < self.stop_distance:
            self.get_logger().error(f"🚨 장애물 감지! (거리: {min(valid_ranges):.2f}m)")
            self.get_logger().info("GUI 팀의 재개 명령을 대기합니다...")
            
            # 1. 상태를 대기로 변경
            self.is_waiting_for_gui = True
            
            # 2. 로봇 즉시 정지명령 발행
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
            
            # 3. GUI 팀에게 경고 메시지 발행
            alert_msg = String()
            alert_msg.data = "OBSTACLE_DETECTED"
            self.alert_pub.publish(alert_msg)

    def nav_callback(self, nav_msg):
        # GUI 허락을 기다리는 중이라면 Nav2 명령 무시하고 속도 0 발행
        if self.is_waiting_for_gui:
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
        else:
            self.cmd_pub.publish(nav_msg)

    def resume_callback(self, msg):
        # GUI 팀에서 보낸 메시지 확인
        if msg.data == "RESUME":
            self.get_logger().info("🟢 GUI 팀으로부터 진행 허락을 받았습니다. 주행을 재개합니다!")
            self.is_waiting_for_gui = False

def main(args=None):
    rclpy.init(args=args)
    node = Robot5SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()