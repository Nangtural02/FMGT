# uwb_publisher_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point # 두 거리 값을 담기 위한 용도로 사용
import serial
import re
import threading
import time

# --- 라즈베리파이 환경에 맞는 설정 ---
SERIAL_PORT_A = '/dev/ttyACM0'
SERIAL_PORT_B = '/dev/ttyACM1'
BAUD_RATE = 115200
DATA_REGEX = re.compile(r"distance\[cm\]=(\d+)")

# SerialReaderThread 클래스는 이전 코드와 동일하게 사용
class SerialReaderThread(threading.Thread):
    # ... (이전과 100% 동일한 코드) ...
    def __init__(self, port, baudrate, regex):
        super().__init__(daemon=True)
        self.port, self.baudrate, self.regex = port, baudrate, regex
        self.latest_distance_m, self.is_running, self.ser = 0.0, True, None
    def run(self):
        while self.is_running:
            try:
                if self.ser is None:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                    # print(f"INFO: {self.port} 연결 성공.") # ROS에서는 logger 사용 권장
                line = self.ser.readline().decode('utf-8').strip()
                if "distance[cm]" in line:
                    match = self.regex.search(line)
                    if match: self.latest_distance_m = int(match.group(1)) / 100.0
            except serial.SerialException:
                if self.ser: self.ser.close()
                self.ser = None
                time.sleep(2)
            except (UnicodeDecodeError, ValueError): continue
            except Exception: pass
    def stop(self):
        self.is_running = False
        if self.ser and self.ser.is_open: self.ser.close()

class UwbPublisherNode(Node):
    def __init__(self):
        super().__init__('uwb_publisher_node')
        self.publisher_ = self.create_publisher(Point, 'raw_uwb_distances', 10)
        
        # 시리얼 리더 스레드 시작
        self.reader_a = SerialReaderThread(SERIAL_PORT_A, BAUD_RATE, DATA_REGEX)
        self.reader_b = SerialReaderThread(SERIAL_PORT_B, BAUD_RATE, DATA_REGEX)
        self.reader_a.start()
        self.reader_b.start()

        # 0.05초 (20Hz) 마다 타이머 콜백 실행
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('UWB Publisher Node가 시작되었습니다.')

    def timer_callback(self):
        msg = Point()
        msg.x = self.reader_a.latest_distance_m # dist_a
        msg.y = self.reader_b.latest_distance_m # dist_b
        msg.z = 0.0 # 사용 안함
        
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing distances: A={msg.x:.2f}, B={msg.y:.2f}') # 디버깅용

    def on_shutdown(self):
        self.get_logger().info('노드 종료... 스레드를 정리합니다.')
        self.reader_a.stop()
        self.reader_b.stop()

def main(args=None):
    rclpy.init(args=args)
    node = UwbPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()