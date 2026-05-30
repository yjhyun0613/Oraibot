import os
import sys
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloNavAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_nav_alert_node')
        self.model = model
        self.bridge = CvBridge()
        # 최신 이미지만 처리하기 위해 크기가 1인 큐 설정
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        
        self.latest_depth = None
        self.nav_paused = False # 로봇의 일시정지 상태를 추적하는 플래그

        # 서비스 및 퍼블리셔 설정
        from nav2_msgs.srv import ManageLifecycleNodes
        # Nav2의 네비게이션 상태를 제어하기 위한 서비스 클라이언트 생성
        self.lifecycle_client = self.create_client(ManageLifecycleNodes, '/robot3/lifecycle_manager_navigation/manage_nodes')
        # 상태 메시지 전송용 퍼블리셔
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        # 감지된 순간의 이미지 전송용 퍼블리셔
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # 토픽 구독 (OAK-D 표준 프리뷰 토픽 사용 시 주석 해제 가능)
        # self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        # self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)

        # 실제 로봇의 압축된 RGB 이미지 토픽 구독
        self.create_subscription(CompressedImage, '/robot3/oakd/rgb/image_raw/compressed', self.rgb_callback, 10)
        # 실제 로봇의 Stereo Depth 이미지 토픽 구독
        self.create_subscription(Image, '/robot3/oakd/stereo/image_raw', self.depth_callback, 10)

        # 메인 루프와 별개로 감지 로직을 수행할 스레드 시작
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Node initialized. Ready for detection.")

    def rgb_callback(self, msg):
        """RGB 이미지 수신 시 호출되는 콜백 함수"""
        try:
            # CompressedImage를 OpenCV에서 사용 가능한 bgr8 형식으로 변환
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is not None:
                # 큐가 가득 찼다면 기존 데이터를 비우고 최신 이미지 삽입
                if self.image_queue.full():
                    try: self.image_queue.get_nowait()
                    except: pass
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Callback Error: {e}")

    def depth_callback(self, msg):
        """Depth 이미지 수신 시 호출되는 콜백 함수"""
        try:
            # 수신된 Depth 정보를 'passthrough' 인코딩으로 변환하여 저장
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth Callback Error: {e}")

    def send_alert_once(self, img, distance):
        """물체가 감지되어 정지하는 순간 알람 메시지와 이미지를 전송하는 함수"""
        msg = String()
        msg.data = f"STOP: Object at {distance:.2f}m"
        self.alert_pub.publish(msg)
        
        try:
            if img is not None and img.size > 0:
                # 1. 원본 이미지의 훼손을 막기 위해 깊은 복사(Deep Copy) 수행
                alert_frame = np.array(img, copy=True)
                
                # 2. 이미지 데이터 형식이 uint8(8비트 부호없는 정수)인지 확인 및 변환
                if alert_frame.dtype != np.uint8:
                    alert_frame = alert_frame.astype(np.uint8)

                # OpenCV 이미지를 ROS 이미지 메시지 형식으로 변환하여 발행
                img_msg = self.bridge.cv2_to_imgmsg(alert_frame, encoding="bgr8")
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn(f"Alert image published! Distance: {distance:.2f}m")
            else:
                self.get_logger().error("Attempted to send an empty image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish alert image: {e}")

    def control_nav(self, command):
        """Nav2의 Lifecycle을 제어하여 네비게이션을 일시정지하거나 재개하는 함수"""
        from nav2_msgs.srv import ManageLifecycleNodes
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error("Nav2 Service not available.")
            return
        req = ManageLifecycleNodes.Request()
        # 명령에 따라 PAUSE(정지) 또는 RESUME(재개) 요청 설정
        req.command = ManageLifecycleNodes.Request.PAUSE if command == "pause" else ManageLifecycleNodes.Request.RESUME
        # 서비스 비동기 호출
        self.lifecycle_client.call_async(req)

    def detection_loop(self):
        """이미지를 분석하고 거리 기반 제어 로직을 수행하는 메인 감지 루프"""
        while not self.should_shutdown:
            try:
                # 큐에서 최신 이미지를 꺼내옴 (데이터가 올 때까지 최대 0.5초 대기)
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            # YOLO 모델을 사용하여 객체 추론 (stream=True를 통해 성능 최적화)
            results = self.model.predict(img, stream=True, verbose=False)
            min_dist = float('inf')
            detected_in_range = False
            depth_img = self.latest_depth

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None: continue
                for box in r.boxes:
                    # 박스의 좌상단(x1, y1) 및 우하단(x2, y2) 좌표 추출
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 바운딩 박스의 중심 좌표 계산
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    current_dist = -1.0
                    if depth_img is not None:
                        # RGB 이미지와 Depth 이미지의 해상도 비율 계산을 통해 좌표 매칭
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx, ty = int(cx * w_d / w_i), int(cy * h_d / h_i)
                        
                        try:
                            # 매칭된 좌표에서 Depth 값(거리) 추출
                            dist_val = depth_img[ty, tx]
                            # 데이터가 mm 단위일 경우 m 단위로 변환
                            current_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)
                            # 유효 범위 내의 거리만 취급 (0.1m ~ 10m)
                            if 0.1 < current_dist < 10.0:
                                min_dist = min(min_dist, current_dist)
                                detected_in_range = True
                        except: pass

                    # 화면에 바운딩 박스 시각화 (빨간색)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if current_dist > 0:
                        dist_text = f"{current_dist:.2f}m"
                        # 거리 정보를 표시 (가독성을 위한 검은색 외곽선 효과와 초록색 글자)
                        cv2.putText(img, dist_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(img, dist_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- 제어 로직 ---
            # 1. 장애물이 1m 이내로 감지되었을 때
            if detected_in_range and min_dist < 1.0:
                if not self.nav_paused:
                    self.get_logger().info("Object detected! Sending alert and pausing Nav.")
                    # 알림 및 이미지를 먼저 전송하고 네비게이션 일시정지 수행
                    self.send_alert_once(img, min_dist)
                    self.control_nav("pause")
                    self.nav_paused = True # 정지 상태로 플래그 변경
            # 2. 장애물이 없거나 거리가 3m 이상으로 확보되었을 때
            elif not detected_in_range or min_dist >= 3.0:
                # 이전에 정지된 상태였다면 주행 재개
                if self.nav_paused:
                    self.get_logger().info("Path clear. Resuming Nav.")
                    self.control_nav("resume")
                    self.nav_paused = False # 주행 상태로 플래그 변경

            # OpenCV 모니터링 창에 결과 출력
            cv2.imshow("Monitor", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break

def main():
    rclpy.init()
    # 지정된 경로에 가중치 파일이 없으면 기본 yolov8n 모델 사용
    model_path = '/home/rokey/rokey_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    node = YoloNavAlertNode(model)
    try:
        # 노드를 실행하면서 콜백 함수들 처리
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 전 생성된 OpenCV 창 닫기 및 자원 해제
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()