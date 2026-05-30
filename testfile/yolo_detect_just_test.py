#!/usr/bin/env python3

import math
import threading
from queue import Queue, Empty

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

from geometry_msgs.msg import PoseStamped
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TaskResult


class YoloGoalResendTestNode(Node):
    def __init__(self):
        super().__init__('yolo_goal_resend_test_node')

        # -------------------------------
        # YOLO / Camera
        # -------------------------------
        model_path = '/home/yoon/project_ws/src/main_proj/yolo/yolo8n_amr_human.pt'
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.image_queue = Queue(maxsize=1)
        self.latest_depth = None
        self.should_shutdown = False

        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/oakd/rgb/preview/depth', self.depth_callback, 10)

        # -------------------------------
        # Navigation
        # -------------------------------
        self.navigator = TurtleBot4Navigator()

        self.goal_active = False
        self.nav_paused = False
        self.last_goal_pose = None
        self.last_goal_name = None

        # 테스트용 목표 좌표
        #   x: -5.825412798897865 y: 0.5146234436951953
        self.test_goal_x = -5.907
        self.test_goal_y = 0.514
        self.test_goal_yaw = 0.0

        # 감지 스레드
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

        # 주행 상태 확인 타이머
        self.create_timer(0.2, self.monitor_nav_status)

        self.get_logger().info('YoloGoalResendTestNode started.')

        # Nav2 활성화 대기 필요하면 사용
        # self.navigator.waitUntilNav2Active()

        # 바로 테스트 goal 출발
        self.send_test_goal()

    # -------------------------------------------------
    # Camera callbacks
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
            self.get_logger().error(f'RGB callback error: {e}')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    # -------------------------------------------------
    # Goal helpers
    # -------------------------------------------------
    def create_pose(self, x, y, yaw_deg):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y

        rad = math.radians(yaw_deg)
        pose.pose.orientation.z = math.sin(rad / 2.0)
        pose.pose.orientation.w = math.cos(rad / 2.0)
        return pose

    def send_test_goal(self):
        goal_pose = self.create_pose(self.test_goal_x, self.test_goal_y, self.test_goal_yaw)
        self.last_goal_pose = goal_pose
        self.last_goal_name = 'TEST_GOAL'

        self.navigator.goToPose(goal_pose)
        self.goal_active = True

        self.get_logger().info(
            f'[GOAL] Sent test goal x={self.test_goal_x:.3f}, '
            f'y={self.test_goal_y:.3f}, yaw={self.test_goal_yaw:.1f}'
        )

    def resend_last_goal(self):
        if self.last_goal_pose is None:
            self.get_logger().warn('[RESEND] No last_goal_pose saved.')
            return

        self.navigator.goToPose(self.last_goal_pose)
        self.goal_active = True

        self.get_logger().info(f'[RESEND] Resent last goal: {self.last_goal_name}')

    def stop_current_goal(self):
        if self.goal_active:
            self.get_logger().warn('[STOP] cancelTask() called due to person event.')
            self.navigator.cancelTask()
        else:
            self.get_logger().info('[STOP] No active goal to cancel.')

    # -------------------------------------------------
    # Nav monitor
    # -------------------------------------------------
    def monitor_nav_status(self):
        if not self.goal_active:
            return

        if self.navigator.isTaskComplete():
            self.goal_active = False
            result = self.navigator.getResult()

            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('[NAV] Goal SUCCEEDED.')

            elif result == TaskResult.CANCELED:
                if self.nav_paused:
                    self.get_logger().warn('[NAV] Goal canceled by person event.')
                else:
                    self.get_logger().warn('[NAV] Goal canceled unexpectedly.')

            else:
                self.get_logger().warn(f'[NAV] Goal finished with result={result}')

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

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if current_dist > 0:
                        dist_text = f'{current_dist:.2f}m'
                        cv2.putText(img, dist_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(img, dist_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # -----------------------------
            # Person event stop / clear resend
            # -----------------------------
            if detected_in_range and min_dist < 3.0:
                if not self.nav_paused:
                    self.get_logger().warn(
                        f'[EVENT] Person detected within 3.0m. min_dist={min_dist:.2f}'
                    )
                    self.nav_paused = True
                    self.stop_current_goal()

            elif not detected_in_range or min_dist >= 5.0:
                if self.nav_paused:
                    self.get_logger().info(
                        f'[CLEAR] Clear condition met. '
                        f'detected_in_range={detected_in_range}, min_dist={min_dist}'
                    )
                    self.nav_paused = False
                    self.resend_last_goal()

            cv2.imshow('Monitor', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_shutdown = True
                break


def main(args=None):
    rclpy.init(args=args)
    node = YoloGoalResendTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.should_shutdown = True
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()