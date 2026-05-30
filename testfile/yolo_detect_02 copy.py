import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from queue import Queue

class YoloDepthEstimator(Node):
    def __init__(self, model):
        super().__init__('yolo_depth_estimator')
        self.model = model
        self.bridge = CvBridge()
        
        # 설정값
        self.ns = '/robot3'  # 사용자님의 네임스페이스
        self.image_queue = Queue(maxsize=1)
        self.depth_image = None
        self.should_shutdown = False

        # 1. RGB 이미지 구독 (CompressedImage 사용)
        self.image_sub = self.create_subscription(
            CompressedImage,
            f'{self.ns}/oakd/rgb/image_raw/compressed',
            self.image_callback,
            10)

        # 2. 뎁스 이미지 구독 (Image 사용)
        self.depth_sub = self.create_subscription(
            Image,
            f'{self.ns}/oakd/stereo/depth/image_raw',
            self.depth_callback,
            10)

        # 검출 루프 실행
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def image_callback(self, msg):
        try:
            # 압축 이미지 해제
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            # 뎁스 이미지는 passthrough로 변환 (mm 단위)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                continue

            if self.depth_image is None:
                continue

            # YOLO 추론
            results = self.model.predict(img, stream=True, verbose=False)

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                
                for box in r.boxes:
                    # 박스 좌표 및 클래스 정보
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label_name = self.model.names[cls]

                    # 1. 박스의 중심점(Center Pixel) 계산
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # 2. 뎁스 맵에서 해당 좌표의 거리값 추출
                    # 뎁스 이미지 크기와 RGB 크기가 다를 수 있으므로 인덱스 확인
                    h_d, w_d = self.depth_image.shape[:2]
                    h_i, w_i = img.shape[:2]
                    
                    # 좌표 매핑 (RGB와 Depth 해상도가 다를 경우 대비)
                    target_x = int(cx * w_d / w_i)
                    target_y = int(cy * h_d / h_i)

                    try:
                        # 단위가 mm일 경우 m로 변환 (/ 1000.0)
                        distance_mm = self.depth_image[target_y, target_x]
                        distance_m = distance_mm / 1000.0
                        
                        if distance_mm > 0:
                            dist_text = f"{distance_m:.2f}m"
                        else:
                            dist_text = "Invalid"
                    except IndexError:
                        dist_text = "Out of range"

                    # 3. 화면 표시
                    label = f"{label_name} {dist_text}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1) # 중심점 표시
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO + Depth Distance", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break

def main():
    # 사용자님의 모델 경로로 수정하세요
    model_path = '/home/rokey/rokey_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    
    rclpy.init()
    model = YOLO(model_path)
    node = YoloDepthEstimator(model)

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()