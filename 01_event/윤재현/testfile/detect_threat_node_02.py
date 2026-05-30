import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

class Robot5SafetyNode(Node):
    def __init__(self):
        super().__init__('robot5_safety_node')
        
        # 주행 속도 관련 Pub/Sub
        self.nav_sub = self.create_subscription(Twist, '/robot5/cmd_vel_nav', self.nav_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        
        # GUI 팀과의 통신
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        self.is_waiting_for_gui = False
        
        # [수정 포인트 1] 거리를 조금 늘려보는 것을 추천합니다 (예: 0.4m = 40cm)
        # 0.3m는 로봇 중심 기준이라 실제 범퍼 기준으로는 거의 닿을락 말락한 거리입니다.
        self.stop_distance = 0.4  
        
        # 감시 범위: -180도(뒤쪽) ~ 90도(왼쪽) 
        # 주의: 이 범위면 오른쪽 앞(-90 ~ 0)은 무시됩니다!
        self.min_angle_rad = math.radians(-180.0)
        self.max_angle_rad = math.radians(0.0)

    def scan_callback(self, msg):
        if self.is_waiting_for_gui:
            return

        obstacle_detected = False
        min_detected_dist = float('inf')

        # [수정 포인트 2] 인덱스 슬라이싱 대신 전체 점을 순회하며 가장 확실하게 검사
        for i, r in enumerate(msg.ranges):
            # 에러 값(inf, NaN)이거나 라이다 스펙을 벗어난 쓰레기값은 무시
            if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
                continue

            # 현재 점의 실제 각도 계산
            raw_angle = msg.angle_min + i * msg.angle_increment
            
            # 어떤 라이다를 쓰든 각도를 무조건 -pi ~ pi (-180 ~ 180도)로 정규화해주는 마법의 공식
            normalized_angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))

            # 현재 점의 각도가 우리가 설정한 감시 구역 안에 있는지 확인
            if self.min_angle_rad <= normalized_angle <= self.max_angle_rad:
                # 구역 안에 있고, 거리가 설정값보다 가깝다면?
                if r < self.stop_distance:
                    obstacle_detected = True
                    if r < min_detected_dist:
                        min_detected_dist = r

        # 장애물이 하나라도 감지되었다면 정지 로직 실행
        if obstacle_detected:
            self.get_logger().error(f"🚨 장애물 감지! (거리: {min_detected_dist:.2f}m)")
            self.get_logger().info("GUI 팀의 재개 명령을 대기합니다...")
            
            self.is_waiting_for_gui = True
            
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
            
            alert_msg = String()
            alert_msg.data = "OBSTACLE_DETECTED"
            self.alert_pub.publish(alert_msg)

    def nav_callback(self, nav_msg):
        if self.is_waiting_for_gui:
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
        else:
            self.cmd_pub.publish(nav_msg)

    def resume_callback(self, msg):
        if msg.data == "RESUME":
            self.get_logger().info("🟢 GUI 팀으로부터 진행 허락을 받았습니다. 주행을 재개합니다!")
            self.is_waiting_for_gui = False

def main(args=None):
    rclpy.init(args=args)
    node = Robot5SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()