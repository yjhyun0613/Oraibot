import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

from tf2_ros import Buffer, TransformListener

class DynamicSafetyNode(Node):
    def __init__(self):
        super().__init__('dynamic_safety_node')
        
        # 설정값
        self.stop_distance = 0.6  # 60cm
        self.is_waiting_for_gui = False
        self.map_data = None
        
        # TF 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 구독자 (Subscribers)
        self.map_sub = self.create_subscription(OccupancyGrid, '/robot5/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/robot5/scan', self.scan_callback, 10)
        self.nav_sub = self.create_subscription(Twist, '/robot5/cmd_vel_nav', self.nav_callback, 10)
        self.resume_sub = self.create_subscription(String, '/robot5/resume_cmd', self.resume_callback, 10)
        
        # 발행자 (Publishers)
        self.cmd_pub = self.create_publisher(Twist, '/robot5/cmd_vel', 10)
        self.alert_pub = self.create_publisher(String, '/robot5/obstacle_alert', 10)
        
        self.get_logger().info("✅ 동적 장애물 감시 노드가 시작되었습니다. (범위: -180~0도)")

    def map_callback(self, msg):
        self.map_data = msg

    def nav_callback(self, nav_msg):
        # 장애물 감지 상태면 정지 명령 발행, 아니면 Nav2 명령 그대로 전달
        if self.is_waiting_for_gui:
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)
        else:
            self.cmd_pub.publish(nav_msg)

    def resume_callback(self, msg):
        if msg.data == "RESUME":
            self.get_logger().info("🟢 주행을 재개합니다.")
            self.is_waiting_for_gui = False

    def scan_callback(self, msg):
        if self.is_waiting_for_gui or self.map_data is None:
            return

        try:
            # base_link에서 map으로의 좌표 변환 정보 가져오기
            trans = self.tf_buffer.lookup_transform('robot5/map', msg.header.frame_id, rclpy.time.Time())
            
            for i, dist in enumerate(msg.ranges):
                angle = msg.angle_min + i * msg.angle_increment
                
                # 사용자가 설정한 감시 범위: -180도 ~ 0도 (오른쪽 및 후방 사각지대)
                if math.radians(-180.0) <= angle <= math.radians(0.0):
                    if msg.range_min < dist < self.stop_distance:
                        
                        # 1. 라이다 점의 로봇 기준 좌표 계산
                        lx = dist * math.cos(angle)
                        ly = dist * math.sin(angle)
                        
                        # 2. 지도 기준 좌표로 변환
                        map_x, map_y = self.transform_to_map(lx, ly, trans)
                        
                        # 3. 지도 데이터와 대조
                        occ_val = self.get_occupancy_value(map_x, map_y)
                        
                        # 4. 지도 값이 0 근처(빈 공간)인데 물체가 있다면 동적 장애물로 판단
                        if 0 <= occ_val < 30:
                            self.get_logger().warn(f"🚨 장애물 감지! 지도상 빈 공간에 물체 확인 (거리: {dist:.2f}m)")
                            self.trigger_stop()
                            break
        except Exception:
            pass

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

    def trigger_stop(self):
        self.is_waiting_for_gui = True
        self.cmd_pub.publish(Twist())
        msg = String()
        msg.data = "OBSTACLE_DETECTED"
        self.alert_pub.publish(msg)

def main():
    rclpy.init()
    node = DynamicSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()