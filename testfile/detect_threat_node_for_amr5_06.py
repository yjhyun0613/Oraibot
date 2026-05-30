# 탐지가 되면 Nav2의 목표를 취소(Cancel)하고, GUI에 알림을 보내며, 사용자가 Rviz에서 새로운 목표를 찍을 때까지 기다리는 안전 노드입니다.
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from action_msgs.srv import CancelGoal
import math

class GoalManagerSafetyNode(Node):
    def __init__(self):
        super().__init__('goal_manager_safety_node')
        
        # --- [검증된 감지 범위] ---
        self.min_distance = 0.20   # 20cm (몸통 무시)
        self.max_distance = 0.60   # 60cm (정지 거리)
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0  
        
        # 상태 변수
        self.is_waiting_for_gui = False
        self.saved_goal = None  # 📍 사용자의 아이디어: 목표를 임시 저장할 변수
        
        # --- [Pub/Sub 설정] ---
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        
        # 🚨 [핵심 1] 목표(Goal) 가로채기 및 재전송용 Pub/Sub
        self.goal_sub = self.create_subscription(PoseStamped, '/robot5/goal_pose', self.goal_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/robot5/goal_pose', 10)
        
        # 🚨 [핵심 2] Nav2 주행 강제 취소(Cancel) 서비스 클라이언트
        self.cancel_client = self.create_client(CancelGoal, '/robot5/navigate_to_pose/_action/cancel_goal')
        
        # 즉시 제동용 조이스틱 토픽
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10)
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🧠 [최종 진화] 목표 저장 & 취소(Cancel) 기반 안전 노드 가동!")
        self.get_logger().info("=========================================")

    def goal_callback(self, msg):
        """Rviz에서 새로운 목표가 들어오면 무조건 저장해 둡니다."""
        self.saved_goal = msg
        self.get_logger().info("📍 새로운 목적지가 안전하게 저장되었습니다.")

    def resume_callback(self, msg):
        """GUI에서 재개(RESUME) 명령이 들어왔을 때"""
        if msg.data == "RESUME":
            self.is_waiting_for_gui = False
            self.get_logger().info("🟢 [GUI] 브레이크 해제!")
            
            # 🚨 저장해둔 목표가 있다면 다시 Nav2로 쏴줍니다!
            if self.saved_goal is not None:
                self.goal_pub.publish(self.saved_goal)
                self.get_logger().info("🚀 임시 저장했던 목적지를 다시 전송하여 주행을 재개합니다!")
            else:
                self.get_logger().warn("⚠️ 저장된 목적지가 없습니다. Rviz에서 새로 찍어주세요.")

    def scan_callback(self, msg):
        if self.is_waiting_for_gui:
            return
            
        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                if min_rad <= angle <= max_rad:
                    self.get_logger().error(f"🚨 갑툭튀 장애물! (거리: {dist:.2f}m) -> 목표 취소 및 정지!")
                    self.trigger_stop()
                    break

    def trigger_stop(self):
        self.is_waiting_for_gui = True
        
        # 1. 즉시 물리적 정지 (브레이크 밟기)
        self.teleop_pub.publish(Twist())
        
        # 2. Nav2 주행 목표 취소 (Nav2 끄기)
        if self.cancel_client.wait_for_service(timeout_sec=1.0):
            req = CancelGoal.Request() # 빈 요청을 보내면 현재 실행 중인 목표가 취소됨
            self.cancel_client.call_async(req)
            self.get_logger().info("🛑 Nav2의 현재 주행 목표를 강제 취소(Cancel)했습니다.")
        else:
            self.get_logger().warn("⚠️ Nav2 취소 서비스를 찾을 수 없습니다.")
            
        # 3. GUI에 알림
        alert = String()
        alert.data = "OBSTACLE_DETECTED"
        self.alert_pub.publish(alert)

def main(args=None):
    rclpy.init(args=args)
    node = GoalManagerSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()