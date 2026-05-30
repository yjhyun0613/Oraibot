import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from action_msgs.srv import CancelGoal
import math
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

from tf2_ros import Buffer, TransformListener

class SmartVisionSafetyNode(Node):
    def __init__(self):
        super().__init__('smart_vision_safety_node')
        
        # --- [1. 감지 설정] ---
        self.min_distance = 0.20   # 20cm
        self.max_distance = 0.60   # 60cm
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0    
        
        # --- [2. 상태 머신 및 저장 변수] ---
        self.state = 'NAVIGATING'
        self.map_data = None
        self.saved_goal = None
        self.latest_image = None
        
        self.get_logger().info("==================================================")
        self.get_logger().info(" 🚀 [1/4] 오차 보정 기능이 탑재된 시스템 초기화...")
        
        # --- [3. YOLO 및 비전 설정] ---
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov8n.pt') 
        self.get_logger().info(" ✅ [YOLO] 모델 로딩 완료!")

        # --- [4. TF (좌표 변환) 설정] ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- [5. ROS 통신 설정 (TurtleBot4 맞춤형)] ---
        self.map_sub = self.create_subscription(OccupancyGrid, '/robot5/map', self.map_callback, 10) 
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/robot5/oakd/rgb/image_raw', self.image_callback, 10)
        
        self.goal_sub = self.create_subscription(PoseStamped, '/robot5/goal_pose', self.goal_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        self.goal_pub = self.create_publisher(PoseStamped, '/robot5/goal_pose', 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10) 
        
        self.cancel_client = self.create_client(CancelGoal, '/robot5/navigate_to_pose/_action/cancel_goal')
        self.timer = self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info(" 🛡️ 똑똑한 동적 장애물 회피 노드 가동 완료!")
        self.get_logger().info("==================================================")

    # ================= 콜백 함수들 =================
    
    def map_callback(self, msg):
        if self.map_data is None:
            self.get_logger().info(" 🗺️ 지도를 수신했습니다! (오차 보정 맵 매칭 준비 완료)")
        self.map_data = msg

    def image_callback(self, msg):
        self.latest_image = msg

    def goal_callback(self, msg):
        self.saved_goal = msg

    def resume_callback(self, msg):
        if msg.data == "RESUME" and self.state == 'WAITING_FOR_GUI':
            self.get_logger().info(" 🟢 [브레이크 해제] 주행을 재개합니다!")
            self.state = 'NAVIGATING'
            if self.saved_goal is not None:
                self.goal_pub.publish(self.saved_goal)

    def scan_callback(self, msg):
        if self.state != 'NAVIGATING' or self.map_data is None:
            return
            
        # 🚨 [수정 포인트 1] TF 에러를 더 이상 숨기지 않고 5초에 한 번씩 터미널에 경고합니다.
        try:
            target_frame = self.map_data.header.frame_id if self.map_data.header.frame_id else 'map'
            trans = self.tf_buffer.lookup_transform(target_frame, msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"⚠️ [TF 에러] 좌표를 못 찾아서 감지를 쉬고 있습니다: {e}", throttle_duration_sec=5.0)
            return 

        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                if min_rad <= angle <= max_rad:
                    lx = dist * math.cos(angle)
                    ly = dist * math.sin(angle)
                    map_x, map_y = self.transform_to_map(lx, ly, trans)
                    
                    # 🚨 [수정 포인트 2] 단순 1픽셀 검사가 아닌 반경 검사 실행
                    if self.is_dynamic_obstacle(map_x, map_y):
                        self.get_logger().error("==================================================")
                        self.get_logger().error(f" 🚨 [장애물 확정] 벽이 아닌 동적 장애물 감지! (거리: {dist:.2f}m)")
                        self.get_logger().error("==================================================")
                        
                        self.trigger_stop()
                        break 

    # ================= 핵심 로직 함수들 =================

    def trigger_stop(self):
        self.teleop_pub.publish(Twist()) 
        if self.cancel_client.wait_for_service(timeout_sec=0.5):
            req = CancelGoal.Request()
            self.cancel_client.call_async(req)
        self.state = 'DETECTING'

    def control_loop(self):
        if self.state == 'DETECTING':
            self.run_yolo_detection()

    def run_yolo_detection(self):
        if self.latest_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            results = self.yolo_model(cv_image, verbose=False)[0]
            
            detected_objects = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                detected_objects.append(class_name)
            
            if detected_objects:
                msg = f"위협 식별됨: {', '.join(set(detected_objects))}"
                self.get_logger().error(f" ⚠️ {msg}")
            else:
                msg = "미확인 장애물"
                self.get_logger().warn(f" ❓ {msg}")
                
            alert = String()
            alert.data = f"OBSTACLE_DETECTED:{msg}"
            self.alert_pub.publish(alert)
            
            self.state = 'WAITING_FOR_GUI'

        except Exception as e:
            self.state = 'WAITING_FOR_GUI'

    # ================= 헬퍼 함수들 (맵 매칭 개선) =================

    def transform_to_map(self, lx, ly, trans):
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        mx = lx * math.cos(yaw) - ly * math.sin(yaw) + trans.transform.translation.x
        my = lx * math.sin(yaw) + ly * math.cos(yaw) + trans.transform.translation.y
        return mx, my

    def is_dynamic_obstacle(self, x, y):
        """
        [핵심 보정 로직]
        점 하나만 보지 않고, 주변 반경(약 15cm)을 모두 훑어봅니다.
        주변에 벽이 하나라도 있으면 위치 오차로 인한 '가짜 장애물(벽)'로 간주하고 무시합니다.
        """
        info = self.map_data.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        # 맵 범위를 벗어나면 검사 불가
        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return False

        # 검색 반경: 3픽셀 (해상도가 0.05m라면 15cm 반경 검사)
        # 벽 근처에서 계속 멈춘다면 이 값을 4나 5로 늘리면 오차에 훨씬 관대해집니다.
        search_radius = 3 
        
        is_free_space = False

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = gx + dx, gy + dy
                
                if 0 <= nx < info.width and 0 <= ny < info.height:
                    idx = ny * info.width + nx
                    val = self.map_data.data[idx]

                    # 주변에 조금이라도 벽(점유율 50 이상)이 묻어 있다면? -> 무조건 벽으로 판정 (동적 장애물 아님)
                    if val >= 50:
                        return False
                        
                    # 주변이 빈 공간(0~30)이거나 미탐사 지역(-1)이면 동적 장애물 후보로 인정
                    if val < 30:
                        is_free_space = True

        # 주변 반경 15cm 내에 벽이 단 하나도 없고 뻥 뚫려있는데 라이다에 뭔가 걸렸다? -> 100% 사람/장애물
        return is_free_space

def main(args=None):
    rclpy.init(args=args)
    node = SmartVisionSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()