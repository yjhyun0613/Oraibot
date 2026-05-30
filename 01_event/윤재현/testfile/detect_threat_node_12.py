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
        self.is_rotating = False  # 🔄 회전 중인지 상태를 체크하는 플래그
        self.saved_goal = None
        
        # --- [Pub/Sub 설정] ---
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        
        # 목표(Goal) 가로채기 및 재전송용 Pub/Sub
        self.goal_sub = self.create_subscription(PoseStamped, '/robot5/goal_pose', self.goal_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/robot5/goal_pose', 10)
        
        # Nav2 주행 강제 취소(Cancel) 서비스 클라이언트
        self.cancel_client = self.create_client(CancelGoal, '/robot5/navigate_to_pose/_action/cancel_goal')
        
        # 즉시 제동 및 회전용 조이스틱 토픽
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10)
        
        self.rotation_timer = None # 회전 루프를 위한 타이머
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🤖 [시선 집중] 목표 취소 & 장애물 방향 회전 안전 노드 가동!")
        self.get_logger().info("=========================================")

    def goal_callback(self, msg):
        """Rviz에서 새로운 목표가 들어오면 무조건 저장해 둡니다."""
        self.saved_goal = msg
        self.get_logger().info("📍 새로운 목적지가 안전하게 저장되었습니다.")

    def resume_callback(self, msg):
        """GUI에서 재개(RESUME) 명령이 들어왔을 때"""
        if msg.data == "RESUME":
            self.is_waiting_for_gui = False
            self.is_rotating = False
            self.get_logger().info("🟢 [GUI] 브레이크 해제!")
            
            # 저장해둔 목표가 있다면 다시 Nav2로 쏴줍니다!
            if self.saved_goal is not None:
                self.goal_pub.publish(self.saved_goal)
                self.get_logger().info("🚀 임시 저장했던 목적지를 다시 전송하여 주행을 재개합니다!")
            else:
                self.get_logger().warn("⚠️ 저장된 목적지가 없습니다. Rviz에서 새로 찍어주세요.")

    def scan_callback(self, msg):
        # GUI 대기 중이거나 이미 회전 중이면 새로운 스캔 데이터를 무시합니다.
        if self.is_waiting_for_gui or self.is_rotating:
            return
            
        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            if self.min_distance < dist < self.max_distance:
                angle_lidar = msg.angle_min + i * msg.angle_increment
                
                if min_rad <= angle_lidar <= max_rad:
                    self.get_logger().error(f"🚨 장애물 감지! (거리: {dist:.2f}m) -> 취소 및 회전 시작!")
                    # 정지 및 회전 로직에 감지된 라이다 각도를 넘겨줍니다.
                    self.trigger_stop_and_rotate(angle_lidar)
                    break

    def trigger_stop_and_rotate(self, angle_lidar):
        """Nav2 취소 후, 계산된 각도만큼 로봇을 회전시킵니다."""
        self.is_rotating = True  # 스캔 콜백 재진입 방지
        
        # 1. 즉시 물리적 정지 (브레이크 밟기)
        self.teleop_pub.publish(Twist())
        
        # 2. Nav2 주행 목표 취소
        if self.cancel_client.wait_for_service(timeout_sec=1.0):
            req = CancelGoal.Request()
            self.cancel_client.call_async(req)
            self.get_logger().info("🛑 Nav2의 현재 주행 목표를 강제 취소했습니다.")
        else:
            self.get_logger().warn("⚠️ Nav2 취소 서비스를 찾을 수 없습니다.")
            
        # 3. 라이다 tf -> 터틀봇 몸통 tf로 변환 (-90도 보정)
        robot_target_angle = angle_lidar - math.radians(90)
        
        # 각도를 -pi ~ pi (-180도 ~ 180도) 사이로 정규화 (보정 후 각도 튐 방지)
        robot_target_angle = math.atan2(math.sin(robot_target_angle), math.cos(robot_target_angle))
        
        # 4. 회전 속도 (rad/s) 및 목표 소요 시간 계산
        # 목표 각도가 양수면 좌회전, 음수면 우회전
        self.angular_speed = 0.5 if robot_target_angle > 0 else -0.5 
        self.rotation_duration = abs(robot_target_angle) / abs(self.angular_speed)
        
        self.get_logger().info(f"🔄 장애물 방향으로 회전합니다! (로봇 기준 목표 각도: {math.degrees(robot_target_angle):.2f}도)")
        
        # 비동기 회전 타이머 시작 (0.1초마다 로봇에게 회전 명령 전송)
        self.rotation_start_time = self.get_clock().now()
        self.rotation_timer = self.create_timer(0.1, self.rotate_loop)

    def rotate_loop(self):
        """타이머에 의해 0.1초마다 호출되어 로봇을 돌리는 함수"""
        elapsed_time = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9
        
        if elapsed_time < self.rotation_duration:
            # 지정된 시간이 다 될 때까지 회전 명령 전송
            twist = Twist()
            twist.angular.z = self.angular_speed
            self.teleop_pub.publish(twist)
        else:
            # 시간 경과 시 회전 완료 처리
            self.teleop_pub.publish(Twist())  # 완전히 정지
            self.rotation_timer.cancel()      # 타이머 종료
            
            self.is_rotating = False
            self.is_waiting_for_gui = True    # GUI 대기 상태 진입
            
            self.get_logger().info("✅ 장애물 방향으로 회전 완료. GUI 입력을 대기합니다.")
            
            # 회전이 완전히 끝난 후 GUI에 알람 전송
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