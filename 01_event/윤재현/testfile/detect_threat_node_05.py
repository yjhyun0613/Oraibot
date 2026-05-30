import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

from tf2_ros import Buffer, TransformListener

class SmartSafetyNode(Node):
    def __init__(self):
        super().__init__('smart_safety_node')
        
        # --- [완벽 검증된 감지 범위] ---
        self.min_distance = 0.20   # 20cm (몸통 무시!)
        self.max_distance = 0.60   # 60cm (여기까지만 감지)
        
        # 원하시는 각도로 수정 가능 (현재 테스트했던 -40 ~ 40도)
        self.min_angle_deg = -180.0 
        self.max_angle_deg = 0.0  
        
        # 상태 변수
        self.is_waiting_for_gui = False
        self.map_data = None
        
        # --- [TF 설정 (지도와 라이다 좌표 맞추기)] ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # --- [Pub/Sub 설정] ---
        # 1. 센서, 지도, 주행 명령 구독
        self.map_sub = self.create_subscription(OccupancyGrid, '/robot5/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.nav_sub = self.create_subscription(Twist, '/robot5/cmd_vel_nav', self.nav_callback, 10)
        
        # 2. GUI 통신
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        
        # 3. 로봇 구동 (일반 주행 + 우선순위 높은 조이스틱 주행)
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        self.teleop_pub = self.create_publisher(Twist, '/robot5/cmd_vel_teleop', 10)
        
        self.get_logger().info("=========================================")
        self.get_logger().info(" 🧠 [스마트 모드] 지도 기반 동적 장애물 회피 가동!")
        self.get_logger().info("=========================================")

    def map_callback(self, msg):
        # 맵 데이터를 성공적으로 받으면 저장
        if self.map_data is None:
            self.get_logger().info("🗺️ 지도를 성공적으로 수신했습니다! (벽 인식 시작)")
        self.map_data = msg

    def nav_callback(self, nav_msg):
        # GUI 대기 중이면 강제 정지
        if self.is_waiting_for_gui:
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
            self.teleop_pub.publish(stop_msg)
        else:
            self.cmd_pub.publish(nav_msg)

    def resume_callback(self, msg):
        if msg.data == "RESUME":
            self.get_logger().info("🟢 [GUI] 주행 재개 승인됨. 다시 움직입니다!")
            self.is_waiting_for_gui = False

    def scan_callback(self, msg):
        # 1. 지도 수신 대기
        if self.map_data is None:
            return
            
        # 2. 좌표 변환 (에러 발생 시 터미널에 빨간 글씨로 띄움!)
        try:
            trans = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().error(f"❌ 좌표 변환 실패! (이것 때문에 감지를 못함): {e}")
            return

        min_rad = math.radians(self.min_angle_deg)
        max_rad = math.radians(self.max_angle_deg)
        
        for i, dist in enumerate(msg.ranges):
            # 3. 라이다가 물체를 감지함 (20~60cm)
            if self.min_distance < dist < self.max_distance:
                angle = msg.angle_min + i * msg.angle_increment
                
                # 4. 각도 안에 들어옴
                if min_rad <= angle <= max_rad:
                    lx = dist * math.cos(angle)
                    ly = dist * math.sin(angle)
                    map_x, map_y = self.transform_to_map(lx, ly, trans)
                    occ_val = self.get_occupancy_value(map_x, map_y)
                    
                    # --- 여기서부터 X-Ray 진단 로그 ---
                    self.get_logger().info(f"👀 라이다는 물체를 봤습니다! (거리: {dist:.2f}m)")
                    
                    if occ_val == -1:
                        self.get_logger().warn(f"🚫 무시됨: 지도값이 {occ_val}(미탐사)입니다. '0~30' 조건에 안 맞습니다!")
                    elif occ_val >= 30:
                        self.get_logger().warn(f"🚫 무시됨: 지도값이 {occ_val}(이미 벽임)입니다. 동적 장애물이 아니라고 판단했습니다!")
                    elif 0 <= occ_val < 30:
                        self.get_logger().error("✅ 완벽한 조건! 여기서 로봇이 멈춰야 합니다!")
                        self.trigger_stop()
                    
                    break # 로그가 너무 도배되지 않게 한 번만 출력하고 스캔 종료

    def transform_to_map(self, lx, ly, trans):
        # 복잡한 행렬 계산 (2D 회전 및 이동)
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        mx = lx * math.cos(yaw) - ly * math.sin(yaw) + trans.transform.translation.x
        my = lx * math.sin(yaw) + ly * math.cos(yaw) + trans.transform.translation.y
        return mx, my

    def get_occupancy_value(self, x, y):
        # x, y 좌표를 맵 데이터 배열의 인덱스로 변환해서 점유율(Occupancy) 값 가져오기
        info = self.map_data.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return self.map_data.data[gy * info.width + gx]
        return -1

    def trigger_stop(self):
        self.is_waiting_for_gui = True
        
        # 즉시 정지 (Teleop 덮어쓰기)
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        self.teleop_pub.publish(stop_msg)
        
        # GUI에 알림
        alert = String()
        alert.data = "OBSTACLE_DETECTED"
        self.alert_pub.publish(alert)

def main(args=None):
    rclpy.init(args=args)
    node = SmartSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()