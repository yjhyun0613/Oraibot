import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from action_msgs.srv import CancelGoal
import math
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

def get_yaw_from_quaternion(q):
    """쿼터니언에서 Yaw 각도(라디안)를 추출하는 헬퍼 함수"""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """각도를 -pi ~ pi 사이로 정규화"""
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

class GoalManagerSafetyNode(Node):
    def __init__(self):
        super().__init__('goal_manager_safety_node')
        
        # --- [검증된 감지 범위] ---
        self.min_distance = 0.20
        self.max_distance = 0.60
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0  
        
        # --- [상태 머신 변수] ---
        # 상태 종류: 'NAVIGATING', 'ROTATING', 'DETECTING', 'WAITING_FOR_GUI'
        self.state = 'NAVIGATING'
        self.saved_goal = None
        self.target_yaw = 0.0
        self.current_yaw = 0.0
        
        # --- [YOLO 및 Vision 설정] ---
        self.bridge = CvBridge()
        self.get_logger().info("⏳ YOLO 모델 로딩 중...")
        # v8, v10, v11 등 사용 중인 모델에 맞게 가중치 파일 이름 변경 가능
        self.yolo_model = YOLO('yolov8n.pt') 
        self.latest_image = None
        self.get_logger().info("✅ YOLO 모델 로딩 완료!")

        # --- [Pub/Sub 설정] ---
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        
        # Odom & Camera 구독 (추가됨)
        self.odom_sub = self.create_subscription(Odometry, '/robot5/odom', self.odom_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/robot5/camera/image_raw', self.image_callback, 10)
        
        # Nav2 제어 Pub/Sub 및 Service Client
        self.goal_sub = self.create_subscription(PoseStamped, '/robot5/goal_pose', self.goal_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/robot5/goal_pose', 10)
        self.cancel_client = self.create_client(CancelGoal, '/robot5/navigate_to_pose/_action/cancel_goal')
        
        # 주행 제어용 토픽
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10)
        
        # 제어 루프 타이머 (주기적인 회전 명령 및 상태 확인용)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🧠 [비전 융합] YOLO 기반 표적 식별 안전 노드 가동!")
        self.get_logger().info("=========================================")

    def goal_callback(self, msg):
        self.saved_goal = msg
        self.get_logger().info("📍 새로운 목적지가 안전하게 저장되었습니다.")

    def resume_callback(self, msg):
        if msg.data == "RESUME" and self.state == 'WAITING_FOR_GUI':
            self.state = 'NAVIGATING'
            self.get_logger().info("🟢 [GUI] 브레이크 해제! 주행 상태로 복귀합니다.")
            if self.saved_goal is not None:
                self.goal_pub.publish(self.saved_goal)
                self.get_logger().info("🚀 임시 저장했던 목적지를 다시 전송하여 주행을 재개합니다!")
            else:
                self.get_logger().warn("⚠️ 저장된 목적지가 없습니다. Rviz에서 새로 찍어주세요.")

    def odom_callback(self, msg):
        """현재 로봇의 회전각(Yaw)을 지속적으로 업데이트"""
        self.current_yaw = get_yaw_from_quaternion(msg.pose.pose.orientation)

    def image_callback(self, msg):
        """가장 최근의 카메라 프레임을 버퍼에 저장"""
        self.latest_image = msg

    def scan_callback(self, msg):
        # 주행 중일 때만 장애물을 스캔합니다.
        if self.state != 'NAVIGATING':
            return
            
        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                if min_rad <= angle <= max_rad:
                    self.get_logger().error(f"🚨 장애물 감지! (거리: {dist:.2f}m) -> 방향 전환 및 탐색 시작!")
                    
                    # 1. Nav2 정지 및 브레이크
                    self.trigger_stop()
                    
                    # 2. 회전 목표 각도 계산 (현재 오돔 기반)
                    self.target_yaw = normalize_angle(self.current_yaw + angle)
                    
                    # 3. 상태 변경 -> 회전 모드 돌입
                    self.state = 'ROTATING'
                    break

    def trigger_stop(self):
        """즉시 제동 및 Nav2 취소"""
        self.teleop_pub.publish(Twist()) # 제동
        
        if self.cancel_client.wait_for_service(timeout_sec=0.5):
            req = CancelGoal.Request()
            self.cancel_client.call_async(req)
            self.get_logger().info("🛑 Nav2 주행 목표 강제 취소 완료.")

    def control_loop(self):
        """상태(State)에 따른 주기적인 동작을 제어합니다."""
        if self.state == 'ROTATING':
            self.rotate_towards_target()
        elif self.state == 'DETECTING':
            self.run_yolo_detection()

    def rotate_towards_target(self):
        """목표 각도로 회전하는 비례 제어(P-Control) 함수"""
        yaw_error = normalize_angle(self.target_yaw - self.current_yaw)
        
        # 오차가 0.08 라디안(약 4.5도) 이내면 회전 완료로 간주
        if abs(yaw_error) < 0.08:
            self.teleop_pub.publish(Twist()) # 회전 정지
            self.get_logger().info("🔄 회전 완료! 카메라 식별 모드로 전환합니다.")
            self.state = 'DETECTING'
            return
            
        # P-제어로 회전 속도 결정
        cmd_msg = Twist()
        cmd_msg.angular.z = 1.0 * yaw_error # P Gain = 1.0
        
        # 최대/최소 속도 제한
        if cmd_msg.angular.z > 0:
            cmd_msg.angular.z = min(max(cmd_msg.angular.z, 0.2), 1.0)
        else:
            cmd_msg.angular.z = max(min(cmd_msg.angular.z, -0.2), -1.0)
            
        self.teleop_pub.publish(cmd_msg)

    def run_yolo_detection(self):
        """로봇이 정지한 후 프레임을 캡처하여 YOLO 추론 진행"""
        if self.latest_image is None:
            self.get_logger().warn("⚠️ 카메라 이미지를 기다리는 중...")
            return

        try:
            # ROS Image -> OpenCV Image 변환
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            # YOLO 객체 탐지 실행 (conf=0.5로 신뢰도 조절 가능)
            results = self.yolo_model(cv_image, verbose=False)[0]
            
            person_detected = False
            for box in results.boxes:
                # COCO 데이터셋 기준 클래스 0이 'person'
                class_id = int(box.cls[0])
                if class_id == 0:
                    person_detected = True
                    break
            
            if person_detected:
                self.get_logger().error("⚠️ [위협 감지] 해당 방향에서 '사람'이 식별되었습니다! 대기 모드로 전환합니다.")
                alert = String()
                alert.data = "PERSON_DETECTED"
                self.alert_pub.publish(alert)
                
                # GUI의 명령을 기다림
                self.state = 'WAITING_FOR_GUI'
            else:
                self.get_logger().info("✅ 감지된 객체는 사람이 아닙니다. 주행을 자동 재개합니다.")
                self.state = 'WAITING_FOR_GUI' # 또는 여기서 즉시 self.state = 'NAVIGATING'으로 전환하여 자동 출발 가능
                
                # 자동 출발을 원할 경우 바로 재개 명령을 호출
                msg = String()
                msg.data = "RESUME"
                self.resume_callback(msg)

        except Exception as e:
            self.get_logger().error(f"비전 처리 중 오류 발생: {e}")
            self.state = 'WAITING_FOR_GUI'

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