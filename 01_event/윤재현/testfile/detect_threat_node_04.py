import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

class FinalSafetyStopNode(Node):
    def __init__(self):
        super().__init__('final_safety_stop_node')
        
        # --- [검증된 파라미터] ---
        self.min_distance = 0.20   # 20cm (몸통 무시)
        self.max_distance = 0.60   # 60cm (감지 거리)
        self.min_angle_deg = -180.0 # 우측 40도
        self.max_angle_deg = 0.0  # 좌측 40도
        
        # 상태 변수
        self.is_waiting_for_gui = False
        
        # --- [Pub/Sub 설정] ---
        # 1. 센서 및 내비게이션 입력 구독
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.nav_sub = self.create_subscription(Twist, '/robot5/cmd_vel_nav', self.nav_callback, 10)
        
        # 2. GUI 통신 (경고 보내기 / 재개 신호 받기)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        # 3. 로봇에게 최종 명령 전달
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🛡️ 터틀봇4 안전 주행 노드 가동 시작")
        self.get_logger().info("=========================================")

    def nav_callback(self, nav_msg):
        """내비게이션 명령을 로봇에게 전달하거나 차단하는 함수"""
        if self.is_waiting_for_gui:
            # 정지 상태면 Nav2 명령 무시하고 속도 0 발행
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
        else:
            # 정상 상태면 Nav2 명령 그대로 로봇에게 전달
            self.cmd_pub.publish(nav_msg)

    def resume_callback(self, msg):
        """GUI 노드로부터 재개 신호를 받는 함수"""
        if msg.data == "RESUME":
            self.get_logger().info("🟢 [GUI] 주행 재개 승인됨. 다시 움직입니다!")
            self.is_waiting_for_gui = False

    def scan_callback(self, msg):
        """라이다 데이터를 분석하여 장애물을 찾는 함수"""
        # 이미 멈춰있으면 계산 중단
        if self.is_waiting_for_gui:
            return
            
        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            # 우리가 성공했던 그 조건문! (20cm ~ 60cm)
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                # 각도 범위 확인
                if min_rad <= angle <= max_rad:
                    self.get_logger().error(f"🚨 장애물 발견! 즉시 정지 (거리: {dist:.2f}m, 각도: {math.degrees(angle):.1f}도)")
                    self.trigger_stop()
                    break 

    def trigger_stop(self):
        """실제로 로봇을 멈추고 경고를 보내는 함수"""
        self.is_waiting_for_gui = True
        
        # 로봇 즉시 정지
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        
        # GUI 팀에게 알람 전송
        alert = String()
        alert.data = "OBSTACLE_DETECTED"
        self.alert_pub.publish(alert)

def main(args=None):
    rclpy.init(args=args)
    node = FinalSafetyStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()