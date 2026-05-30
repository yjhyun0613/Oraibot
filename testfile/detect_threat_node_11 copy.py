"""
=============================================================================
[노드명] smart_vision_safety_node
[작성자] Hanyang Univ. Robotics Team
[설명] 
TurtleBot4(AMR)의 자율주행(Nav2) 중, 정적 지도(Map)에 없는 '동적 장애물(갑툭튀 사람 등)'이
나타났을 때만 즉시 제동하고, OAK-D 카메라와 YOLOv8을 통해 장애물의 정체를 식별하는 안전 제어 노드.

[업데이트 내역]
- Map 수신 오류 해결: Transient Local QoS 프로필 적용
- TF 변환 오류 해결: Namespace 격리 해제 및 글로벌 /tf 참조, 센서 Frame ID 강제 매칭
=============================================================================
"""

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
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

class SmartVisionSafetyNode(Node):
    def __init__(self):
        # 🚨 [수정 1] 방음벽(namespace) 해제! 이제 글로벌 /tf 를 정상적으로 듣습니다.
        super().__init__('smart_vision_safety_node')
        
        # --- [1. 감지 설정] ---
        self.min_distance = 0.20   
        self.max_distance = 0.60   
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0    
        
        self.state = 'NAVIGATING'
        self.map_data = None
        self.saved_goal = None
        self.latest_image = None
        
        self.get_logger().info("==================================================")
        self.get_logger().info(" 🚀 [1/4] 오차 보정 기능 탑재 시스템 초기화...")
        
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov8n.pt') 
        self.get_logger().info(" ✅ [2/4] [YOLO] 모델 로딩 완료!")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # 🚨 [수정 2] 방음벽을 없앴으니, 토픽 이름 앞에 다시 절대 경로('/robot5/')를 붙여줍니다.
        self.map_sub = self.create_subscription(OccupancyGrid, '/robot5/map', self.map_callback, map_qos) 
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
            
        try:
            target_frame = 'robot5/map'
            
            # 🚨 [수정 3] 라이다 데이터 이름표에 강제로 'robot5/'를 붙여 뼈대 이름과 짝을 맞춰줍니다.
            source_frame = msg.header.frame_id
            if not source_frame.startswith('robot5/'):
                source_frame = 'robot5/' + source_frame
                
            trans = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"⚠️ [TF 에러] 좌표 변환 대기 중... : {e}", throttle_duration_sec=5.0)
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

    # ================= 헬퍼 함수들 =================
    def transform_to_map(self, lx, ly, trans):
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        mx = lx * math.cos(yaw) - ly * math.sin(yaw) + trans.transform.translation.x
        my = lx * math.sin(yaw) + ly * math.cos(yaw) + trans.transform.translation.y
        return mx, my

    def is_dynamic_obstacle(self, x, y):
        info = self.map_data.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        if not (0 <= gx < info.width and 0 <= gy < info.height):
            return False

        search_radius = 3 
        is_free_space = False

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = gx + dx, gy + dy
                
                if 0 <= nx < info.width and 0 <= ny < info.height:
                    idx = ny * info.width + nx
                    val = self.map_data.data[idx]

                    if val >= 50:
                        return False
                        
                    if val < 30:
                        is_free_space = True

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