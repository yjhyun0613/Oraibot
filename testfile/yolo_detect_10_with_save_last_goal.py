# 핵심 아이디어는 이겁니다.

# 마지막 navigation goal을 저장
# 사람이 가까워지면 pause
# 사람이 사라지면 resume만 믿지 않고
# 저장해둔 last goal을 다시 NavigateToPose로 재전송

import os
import rclpy
import threading
from queue import Queue, Empty

from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes

from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2
import numpy as np


class YoloNavAlertNode(Node):
    def __init__(self, model):
        super().__init__('yolo_nav_alert_node')

        self.model = model
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False

        self.latest_depth = None
        self.nav_paused = False

        # 마지막 목표 저장
        self.last_goal_pose = None

        # 마지막으로 재전송한 goal handle/result 추적
        self.current_goal_handle = None
        self.current_result_future = None

        # Nav2 lifecycle pause/resume
        self.lifecycle_client = self.create_client(
            ManageLifecycleNodes,
            '/lifecycle_manager_navigation/manage_nodes'
        )

        # Nav2 goal 재전송용 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Alert publishers
        self.alert_pub = self.create_publisher(String, '/yolo/alert_status', 10)
        self.alert_img_pub = self.create_publisher(Image, '/yolo/alert_image', 10)

        # Camera subscriptions
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)

        # 마지막 goal 저장용 subscription
        # 실제 시스템 goal 토픽 이름에 맞게 필요하면 수정
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

        self.get_logger().info('YoloNavAlertNode initialized.')

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if img is not None:
                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()
                    except Exception:
                        pass
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f'RGB Callback Error: {e}')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth Callback Error: {e}')

    def goal_callback(self, msg):
        self.last_goal_pose = msg
        self.get_logger().info(
            f'Last goal saved: '
            f'x={msg.pose.position.x:.3f}, '
            f'y={msg.pose.position.y:.3f}, '
            f'z={msg.pose.position.z:.3f}'
        )

    # -------------------------------------------------
    # Alert / Nav control
    # -------------------------------------------------
    def send_alert_once(self, img, distance):
        msg = String()
        msg.data = f'STOP: Object at {distance:.2f}m'
        self.alert_pub.publish(msg)

        try:
            if img is not None and img.size > 0:
                alert_frame = np.array(img, copy=True)

                if alert_frame.dtype != np.uint8:
                    alert_frame = alert_frame.astype(np.uint8)

                img_msg = self.bridge.cv2_to_imgmsg(alert_frame, encoding='bgr8')
                self.alert_img_pub.publish(img_msg)
                self.get_logger().warn(f'Alert image published! Distance: {distance:.2f}m')
            else:
                self.get_logger().error('Attempted to send an empty image.')
        except Exception as e:
            self.get_logger().error(f'Failed to publish alert image: {e}')

    def control_nav(self, command: str):
        if not self.lifecycle_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error('Nav2 Service not available.')
            return

        req = ManageLifecycleNodes.Request()
        req.command = (
            ManageLifecycleNodes.Request.PAUSE
            if command == 'pause'
            else ManageLifecycleNodes.Request.RESUME
        )

        future = self.lifecycle_client.call_async(req)
        future.add_done_callback(
            lambda f: self.lifecycle_response_callback(f, command)
        )

        self.get_logger().info(f'Lifecycle command sent: {command}')

    def lifecycle_response_callback(self, future, command):
        try:
            _ = future.result()
            self.get_logger().info(f'Lifecycle {command} response received.')
        except Exception as e:
            self.get_logger().error(f'Lifecycle {command} failed: {e}')

    # -------------------------------------------------
    # Goal resend
    # -------------------------------------------------
    def resend_last_goal(self):
        if self.last_goal_pose is None:
            self.get_logger().warn('No saved last goal. Cannot resend.')
            return

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('NavigateToPose action server not available.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.last_goal_pose

        self.get_logger().info(
            f'Resending last goal: '
            f'x={self.last_goal_pose.pose.position.x:.3f}, '
            f'y={self.last_goal_pose.pose.position.y:.3f}, '
            f'z={self.last_goal_pose.pose.position.z:.3f}'
        )

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().warn('Resent goal was rejected.')
                self.current_goal_handle = None
                self.current_result_future = None
                return

            self.get_logger().info('Resent goal accepted.')
            self.current_goal_handle = goal_handle
            self.current_result_future = goal_handle.get_result_async()
            self.current_result_future.add_done_callback(self.goal_result_callback)

        except Exception as e:
            self.get_logger().error(f'Goal response callback error: {e}')

    def goal_result_callback(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'Resent goal finished. status={result.status}')
        except Exception as e:
            self.get_logger().error(f'Goal result callback error: {e}')

    # -------------------------------------------------
    # Detection loop
    # -------------------------------------------------
    def detection_loop(self):
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except Empty:
                continue

            results = self.model.predict(img, stream=True, verbose=False)

            min_dist = float('inf')
            detected_in_range = False
            depth_img = self.latest_depth

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    current_dist = -1.0

                    if depth_img is not None:
                        h_d, w_d = depth_img.shape[:2]
                        h_i, w_i = img.shape[:2]
                        tx = int(cx * w_d / w_i)
                        ty = int(cy * h_d / h_i)

                        try:
                            dist_val = depth_img[ty, tx]
                            current_dist = dist_val / 1000.0 if dist_val > 100 else float(dist_val)

                            if 0.1 < current_dist < 10.0:
                                min_dist = min(min_dist, current_dist)
                                detected_in_range = True

                        except Exception:
                            pass

                    # visualize
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if current_dist > 0:
                        dist_text = f'{current_dist:.2f}m'
                        cv2.putText(
                            img, dist_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3
                        )
                        cv2.putText(
                            img, dist_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )

            # -------------------------------
            # Pause / Resume + Goal Resend
            # -------------------------------
            if detected_in_range and min_dist < 3.0:
                if not self.nav_paused:
                    self.get_logger().info(
                        f'[EVENT] Object detected within threshold. '
                        f'min_dist={min_dist:.2f}m -> pause'
                    )
                    self.send_alert_once(img, min_dist)
                    self.control_nav('pause')
                    self.nav_paused = True
                    self.get_logger().info('[EVENT] nav_paused set to True')

            elif not detected_in_range or min_dist >= 5.0:
                if self.nav_paused:
                    self.get_logger().info(
                        f'[RESUME] Clear condition met. '
                        f'detected_in_range={detected_in_range}, min_dist={min_dist}'
                    )

                    self.control_nav('resume')
                    self.nav_paused = False
                    self.get_logger().info('[RESUME] nav_paused set to False')

                    # resume 뒤 원래 goal 재전송
                    self.resend_last_goal()

            # Monitor
            cv2.imshow('Monitor', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break


def main():
    rclpy.init()

    model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')

    node = YoloNavAlertNode(model)

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()