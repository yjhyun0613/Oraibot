import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from nav2_msgs.msg import SpeedLimit
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class HumanSafetyController(Node):
    def __init__(self):
        super().__init__('human_safety_controller')
        
        # 1. YOLO 모델 로드 (Nano 모델 권장)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # 2. 구독 및 발행 설정
        # OAK-D 카메라의 RGB와 Depth 토픽 (터틀봇4 기본값 확인 필요)
        self.image_sub = self.create_subscription(Image, '/robot3/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/robot3/oakd/stereo/depth/image_raw', self.depth_callback, 10)
        
        # Nav2의 속도 제한 토픽
        self.speed_limit_pub = self.create_publisher(SpeedLimit, '/robot3/speed_limit', 10)

        self.latest_depth_map = None
        self.max_speed = 0.5  # 로봇의 평소 최대 속도 (m/s)

    def depth_callback(self, msg):
        # 뎁스 이미지를 넘파이 배열로 변환 (단위: mm)
        self.latest_depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        if self.latest_depth_map is None:
            return

        # RGB 이미지 변환 및 YOLO 추론
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, classes=[0], verbose=False) # class 0은 'person'

        min_dist = 10.0 # 기본값 (멀리 있음)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box 중심점 계산
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 뎁스 맵에서 중심점 거리 추출 (mm -> m 변환)
                # 0값은 측정 오류이므로 제외
                dist_mm = self.latest_depth_map[cy, cx]
                if dist_mm > 0:
                    dist_m = dist_mm / 1000.0
                    min_dist = min(min_dist, dist_m)

        self.apply_speed_limit(min_dist)

    def apply_speed_limit(self, distance):
        limit_msg = SpeedLimit()
        
        # 거리별 감속 로직
        if distance <= 0.3:
            limit_val = 0.0      # 30cm 이내면 정지
        elif distance <= 1.0:
            # 1m~0.3m 사이에서 선형 감속
            scale = (distance - 0.3) / (1.0 - 0.3)
            limit_val = self.max_speed * scale
        else:
            limit_val = self.max_speed # 감속 없음

        # Nav2 SpeedLimit 메시지 설정 (percentage=False일 경우 실제 속도 값)
        limit_msg.speed_limit = limit_val
        limit_msg.percentage = False 
        self.speed_limit_pub.publish(limit_msg)
        
        if limit_val < self.max_speed:
            self.get_logger().info(f"사람 감지! 거리: {distance:.2f}m -> 제한속도: {limit_val:.2f}m/s")

def main():
    rclpy.init()
    node = HumanSafetyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()