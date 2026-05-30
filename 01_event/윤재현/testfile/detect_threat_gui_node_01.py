import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

class TerminalGUINode(Node):
    def __init__(self):
        super().__init__('terminal_gui_node')
        
        # 1. 로봇의 경고 받기
        self.alert_sub = self.create_subscription(
            String, '/robot5/obstacle_alert', self.alert_callback, 10)
        
        # 2. 로봇에게 재개 명령 보내기
        self.resume_pub = self.create_publisher(String, '/robot5/resume_cmd', 10)
        
        self.get_logger().info("🖥️ 관제(GUI) 터미널 시작. 로봇의 상태를 감시합니다.")
        
        # 사용자 키보드 입력을 백그라운드에서 기다리기 위한 스레드 시작
        self.input_thread = threading.Thread(target=self.wait_for_user_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    def alert_callback(self, msg):
        if msg.data == "OBSTACLE_DETECTED":
            print("\n" + "="*50)
            print(" 🚨 [긴급] 로봇이 장애물을 감지하고 정지했습니다!")
            print(" 🚨 카메라나 현장을 확인하세요.")
            print(" 👉 다시 주행시키려면 'r'을 입력하고 Enter를 누르세요.")
            print("="*50 + "\n")

    def wait_for_user_input(self):
        while rclpy.ok():
            # 터미널에서 사용자 입력 대기
            user_input = input()
            
            # 'r'을 입력했다면
            if user_input.strip().lower() == 'r':
                print("▶️ 로봇에게 [주행 재개] 신호를 전송했습니다.\n")
                
                # 로봇에게 보낼 메시지 생성 및 전송
                msg = String()
                msg.data = "RESUME"
                self.resume_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TerminalGUINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()