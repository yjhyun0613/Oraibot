import os
import sys
import rclpy
import threading
from queue import Queue
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path
import cv2

class YoloDepthNode(Node):
    def __init__(self, model):
        super().__init__('yolo_depth_node')
        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']
        
        # 뎁스 이미지를 저장할 변수
        self.latest_depth = None

        # 로봇 네임스페이스
        ns = '/robot3'

        # 1. RGB 이미지 구독 (CompressedImage)
        self.rgb_subscription = self.create_subscription(
            CompressedImage,
            f'{ns}/oakd/rgb/image_raw/compressed',
            self.rgb_callback,
            10)

        # 2. 뎁스 이미지 구독 (Image)
        self.depth_subscription = self.create_subscription(
            CompressedImage,
            f'{ns}/oakd/stereo/image_raw/compressed',
            self.depth_callback,
            10)

        # YOLO 디텍션 쓰레드 시작
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

    def rgb_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            # 뎁스 이미지를 파이썬 배열로 변환
            self.latest_depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                continue

            # 처리 중 뎁스 이미지가 변경되는 것을 방지하기 위해 복사
            depth_img = self.latest_depth

            # YOLO 모델 추론
            results = self.model.predict(img, stream=True, verbose=False)

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue
                
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0

                    # 바운딩 박스의 중심점 (u, v) 계산
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    distance_str = "N/A"
                    
                    if depth_img is not None:
                        # RGB 이미지와 Depth 이미지의 해상도가 다를 경우를 위한 비율 계산
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]

                        target_x = int(cx * w_d / w_i)
                        target_y = int(cy * h_d / h_i)

                        # 이미지 인덱스 초과 방지
                        if 0 <= target_x < w_d and 0 <= target_y < h_d:
                            distance = depth_img[target_y, target_x]
                            
                            if distance > 0:
                                # 시뮬레이터/실제 카메라에 따라 단위(mm 또는 m)가 다를 수 있음
                                if distance > 100:  # mm 단위로 들어올 경우
                                    distance_m = distance / 1000.0
                                else:               # m 단위로 들어올 경우
                                    distance_m = float(distance)
                                distance_str = f"{distance_m:.2f}m"
                            else:
                                distance_str = "0.0m"

                    label = f"{self.classNames[cls]} {conf:.2f} | Dist: {distance_str}"
                    
                    # 화면에 박스, 중심점, 텍스트 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 + Depth Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via 'q'")
                self.should_shutdown = True
                break

def main():
    # 사용하시는 커스텀 모델 경로
    model_path = '/home/rokey/rokey_ws/src/main_proj/yolo/yolo8n_amr_human.pt'

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        print("Fallback: Using default 'yolov8n.pt'")
        model_path = 'yolov8n.pt' # 파일이 없으면 기본 모델 사용

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"Unsupported model format: {suffix}")
        sys.exit(1)

    rclpy.init()
    node = YoloDepthNode(model)

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()