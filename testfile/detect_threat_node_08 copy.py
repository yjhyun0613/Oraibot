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
        self.min_distance = 0.20   
        self.max_distance = 0.60   
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0    
        
        # --- [2. 상태 머신 및 저장 변수] ---
        self.state = 'NAVIGATING'
        self.map_data = None
        self.saved_goal = None
        self.latest_image = None
        
        self.get_logger().info("==================================================")
        self.get_logger().info(" 🚀 [1/4] 시스템 초기화 시작...")
        self.get_logger().info(f" ⚙️  [설정값] 감지 범위: {self.min_distance}m ~ {self.max_distance}m")
        self.get_logger().info(f" ⚙️  [설정값] 감지 각도: {self.min_angle_deg}도 ~ {self.max_angle_deg}도")
        
        # --- [3. YOLO 및 비전 설정] ---
        self.bridge = CvBridge()
        self.get_logger().info(" ⏳ [2/4] YOLOv8 모델을 메모리에 로딩하고 있습니다...")
        self.yolo_model = YOLO('yolov8n.pt') 
        self.get_logger().info(" ✅ [YOLO] 모델 로딩 완료!")

        # --- [4. TF (좌표 변환) 설정] ---
        self.get_logger().info(" ⏳ [3/4] TF 버퍼(좌표 변환기)를 생성합니다...")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- [5. ROS 통신 설정] ---
        self.get_logger().info(" ⏳ [4/4] ROS 통신 토픽들을 연결합니다...")
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10) 
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/robot5/camera/image_raw', self.image_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/robot5/goal_pose', self.goal_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        self.goal_pub = self.create_publisher(PoseStamped, '/robot5/goal_pose', 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10) 
        
        self.cancel_client = self.create_client(CancelGoal, '/robot5/navigate_to_pose/_action/cancel_goal')
        
        self.timer = self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info("==================================================")
        self.get_logger().info(" 🛡️ 동적 장애물 회피 & 비전 식별 노드 가동 완료!")
        self.get_logger().info(f" 🔄 현재 상태: {self.state} (주행 모드)")
        self.get_logger().info("==================================================")

    # ================= 콜백 함수들 =================
    
    def map_callback(self, msg):
        if self.map_data is None:
            self.get_logger().info(f" 🗺️ [데이터 수신] 지도를 성공적으로 받았습니다! (해상도: {msg.info.resolution:.2f}m/px)")
            self.get_logger().info(" 🔍 이제부터 지도와 라이다를 대조하여 '벽'과 '움직이는 물체'를 구분합니다.")
        self.map_data = msg

    def image_callback(self, msg):
        # 이미지는 초당 수십 장씩 들어오므로, 처음 들어왔을 때만 로그를 남깁니다.
        if self.latest_image is None:
            self.get_logger().info(" 📷 [데이터 수신] 카메라 프레임이 정상적으로 들어오고 있습니다.")
        self.latest_image = msg

    def goal_callback(self, msg):
        self.saved_goal = msg
        self.get_logger().info(f" 📍 [목표 수신] Nav2 목표지점 가로채기 완료! (X: {msg.pose.position.x:.2f}, Y: {msg.pose.position.y:.2f})")

    def resume_callback(self, msg):
        if msg.data == "RESUME":
            self.get_logger().info(" 📩 [통신 수신] GUI 팀으로부터 'RESUME(재개)' 명령을 받았습니다.")
            
            if self.state == 'WAITING_FOR_GUI':
                self.get_logger().info(" 🟢 [상태 변경] WAITING_FOR_GUI -> NAVIGATING (브레이크 해제)")
                self.state = 'NAVIGATING'
                
                if self.saved_goal is not None:
                    self.get_logger().info(" 🚀 [명령 전송] 저장해둔 목적지로 Nav2 주행을 다시 시작합니다!")
                    self.goal_pub.publish(self.saved_goal)
                else:
                    self.get_logger().warn(" ⚠️ [경고] 저장된 목적지가 없습니다. Rviz에서 새로 찍어주세요.")
            else:
                self.get_logger().info(f" ℹ️ [무시됨] 현재 상태가 '{self.state}'이므로 재개 명령을 무시합니다.")

    def scan_callback(self, msg):
        if self.state != 'NAVIGATING' or self.map_data is None:
            return
            
        try:
            trans = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            # TF 오류가 너무 도배되지 않도록 제어
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
                    occ_val = self.get_occupancy_value(map_x, map_y)
                    
                    if 0 <= occ_val < 50:
                        # 🚨 여기서 장애물 감지 프로세스가 시작됩니다!
                        self.get_logger().error("==================================================")
                        self.get_logger().error(f" 🚨 [장애물 감지] 라이다가 물체를 발견했습니다!")
                        self.get_logger().error(f"    - 거리: {dist:.2f}m, 각도: {math.degrees(angle):.1f}도")
                        self.get_logger().error(f"    - 지도 데이터 값(occ_val): {occ_val} (빈 공간임이 확인됨!)")
                        self.get_logger().error("==================================================")
                        
                        self.trigger_stop()
                        break 

    # ================= 핵심 로직 함수들 =================

    def trigger_stop(self):
        self.get_logger().info(" 🛑 [1/3] 강제 제동 명령(cmd_vel_teleop)을 발행합니다.")
        self.teleop_pub.publish(Twist()) 
        
        self.get_logger().info(" 🛑 [2/3] Nav2 주행 목표(CancelGoal)를 찢어버립니다.")
        if self.cancel_client.wait_for_service(timeout_sec=0.5):
            req = CancelGoal.Request()
            self.cancel_client.call_async(req)
            self.get_logger().info(" ✅ [Nav2 취소] 기존 주행 경로가 완전히 취소되었습니다.")
        else:
            self.get_logger().warn(" ⚠️ [Nav2 취소 실패] Cancel 서비스에 연결할 수 없습니다.")

        self.get_logger().info(" 🔄 [3/3] [상태 변경] NAVIGATING -> DETECTING (비전 탐지 모드 돌입)")
        self.state = 'DETECTING'

    def control_loop(self):
        if self.state == 'DETECTING':
            self.run_yolo_detection()

    def run_yolo_detection(self):
        if self.latest_image is None:
            self.get_logger().warn(" ⚠️ [YOLO 대기 중] 카메라 프레임을 기다리고 있습니다...", throttle_duration_sec=1.0)
            return

        try:
            self.get_logger().info(" 👀 [YOLO 분석] 카메라 이미지를 추출하여 분석을 시작합니다.")
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            results = self.yolo_model(cv_image, verbose=False)[0]
            
            detected_objects = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                detected_objects.append(class_name)
            
            self.get_logger().info(f" 🔍 [YOLO 결과] 화면에서 {len(detected_objects)}개의 객체를 찾았습니다.")
            
            if detected_objects:
                unique_objects = ', '.join(set(detected_objects))
                msg = f"위협 식별됨: {unique_objects}"
                self.get_logger().error(f" ⚠️ [위협 판별] {msg}")
            else:
                msg = "미확인 장애물 (객체 인식 안됨)"
                self.get_logger().warn(f" ❓ [위협 판별] {msg}")
                
            self.get_logger().info(" 📡 [GUI 알림] GUI 팀에게 장애물 발견 알람을 전송합니다.")
            alert = String()
            alert.data = f"OBSTACLE_DETECTED:{msg}"
            self.alert_pub.publish(alert)
            
            self.get_logger().info(" 🔄 [상태 변경] DETECTING -> WAITING_FOR_GUI (대기 모드 돌입)")
            self.get_logger().info(" 💤 GUI에서 진행 명령(RESUME)을 내려줄 때까지 제자리에 대기합니다.")
            self.state = 'WAITING_FOR_GUI'

        except Exception as e:
            self.get_logger().error(f" ❌ [비전 처리 오류] {e}")
            self.state = 'WAITING_FOR_GUI'

    # ================= 헬퍼 함수들 =================

    def transform_to_map(self, lx, ly, trans):
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        mx = lx * math.cos(yaw) - ly * math.sin(yaw) + trans.transform.translation.x
        my = lx * math.sin(yaw) + ly * math.cos(yaw) + trans.transform.translation.y
        return mx, my

    def get_occupancy_value(self, x, y):
        info = self.map_data.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.map_data.data[gy * info.width + gx]
        return -1

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