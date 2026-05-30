import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class LidarQuadrantFilter(Node):
    def __init__(self):
        super().__init__('lidar_quadrant_filter')
        # 원본 스캔 구독
        self.subscription = self.create_subscription(
            LaserScan,
            'robot5/scan', 
            self.scan_callback,
            10)
        # 필터링된 스캔 발행
        self.publisher = self.create_publisher(LaserScan, 'robot5/scan_filtered', 10)

    def scan_callback(self, msg):
        # LaserScan 메시지는 ranges가 튜플 형태이므로 리스트로 변환
        filtered_ranges = list(msg.ranges)

        for i in range(len(filtered_ranges)):
            # 현재 인덱스의 각도 계산 (라디안)
            angle = msg.angle_min + i * msg.angle_increment

            # 각도를 0 ~ 2*pi 범위로 정규화 (RPLidar 설정에 따라 다를 수 있음 대비)
            normalized_angle = angle % (2 * math.pi)

            # 1사분면 (0도 ~ 90도, 즉 0 ~ pi/2)에 해당하는지 확인
            if 0.0 <= normalized_angle <= (math.pi / 2.0):
                # 해당 구역의 데이터 값을 무한대(inf)로 설정하여 무시
                filtered_ranges[i] = float('inf') 

        # 수정된 배열을 원본 메시지에 덮어씌움
        msg.ranges = filtered_ranges
        
        # SLAM이나 Nav2에서 사용할 수 있도록 재발행
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarQuadrantFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()