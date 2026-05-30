import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import cv2

class HumanSafetyFilter(Node):
    def __init__(self):
        super().__init__('human_safety_filter')
        self.bridge = CvBridge()
        
        # 1. 사용자님 환경에 맞는 네임스페이스 설정
        self.ns = '/robot5'
        
        # 2. YOLO 모델 로드 (파일 경로 확인 필요)
        self.model = YOLO("yolov8n.pt") 

        # 3. 구독/발행 설정
        self.create_subscription(CameraInfo, f'{self.ns}/oakd/rgb/preview/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, f'{self.ns}/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, f'{self.ns}/oakd/stereo/depth/image_raw', self.depth_callback, 10)
        
        # Nav2의 속도를 받아서 가공 후 실제 로봇에게 전달
        self.nav_vel_sub = self.create_subscription(Twist, f'{self.ns}/cmd_vel_nav', self.nav_vel_callback, 10)
        self.real_vel_pub = self.create_publisher(Twist, f'{self.ns}/cmd_vel', 10)

        # 변수 초기화
        self.K = None
        self.depth_image = None
        self.current_min_dist = 10.0
        self.STOP_DIST = 0.5  # 50cm 이내면 정지
        self.SLOW_DIST = 1.0  # 1m 이내면 감속 시작

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
    
    def rgb_callback(self, msg):
        if self.K is None or self.depth_image is None: return
        
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, verbose=False, classes=[0])[0] # 0: person

        tmp_min_dist = 10.0
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            u, v = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 뎁스 이미지에서 거리 추출 (단위 변환 주의: mm -> m)
            z = float(self.depth_image[v, u]) / 1000.0 if self.depth_image[v, u] > 0 else 10.0
            if z < tmp_min_dist: tmp_min_dist = z
            
            # 화면 표시용
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{z:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.current_min_dist = tmp_min_dist
        cv2.imshow("Safety Monitor", frame)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def nav_vel_callback(self, nav_msg):
        final_vel = Twist()
        scale = 1.0

        # 거리 기반 scale 계산
        if self.current_min_dist <= self.STOP_DIST:
            scale = 0.0
        elif self.current_min_dist <= self.SLOW_DIST:
            # (현재거리 - 정지거리) / (감속시작거리 - 정지거리)
            scale = (self.current_min_dist - self.STOP_DIST) / (self.SLOW_DIST - self.STOP_DIST)
        
        final_vel.linear.x = nav_msg.linear.x * scale
        final_vel.angular.z = nav_msg.angular.z * scale
        
        self.real_vel_pub.publish(final_vel)

def main():
    rclpy.init()
    node = HumanSafetyFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()