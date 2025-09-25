# fmgt6/cli_publisher_node.py
import rclpy
import serial
import re
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header

SERIAL_PORT_A = '/dev/ttyACM0' #left(blue)
SERIAL_PORT_B = '/dev/ttyACM1' #right(red)
BAUD_RATE = 115200
DATA_REGEX = re.compile(r"distance\[cm\]=(\d+)")

class SerialReaderThread(threading.Thread):
    def __init__(self, port, baudrate, regex, logger):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.regex = regex
        self._logger = logger
        
        self.latest_distance_m = 0.0
        self.is_running = True
        self.ser = None

    def run(self):
        # 스레드가 종료될 때 시리얼 포트가 열려 있으면 닫도록 보장합니다.
        try:
            while self.is_running and rclpy.ok():
                try:
                    # 연결이 끊겼을 때만 새로 연결 시도
                    if self.ser is None:
                        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                        self._logger.info(f"Connection with {self.port} Success.")

                    # 데이터를 읽고 파싱
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        match = self.regex.search(line)
                        if match:
                            self.latest_distance_m = int(match.group(1)) / 100.0
                
                except serial.SerialException:
                    if self.ser:
                        self.ser.close()
                    self.ser = None
                    # is_running 플래그가 여전히 True일 때만 재시도 메시지 출력
                    if self.is_running:
                        self._logger.warn(f"{self.port} Disconnected. Retry after 2 second...")
                        time.sleep(2)
                
                except (ValueError, AttributeError) as e:
                    self._logger.warn(f"Data parsing error in {self.port} : {e}")
                    continue
        
        finally:
            # 스레드가 종료될 때 리소스를 스스로 정리합니다.
            if self.ser and self.ser.is_open:
                self.ser.close()
                self._logger.info(f"Connection in {self.port} was successfully finished")

    def stop(self):
        # 스레드에 종료 신호만 보냅니다. 리소스 정리는 run() 메소드가 직접 합니다.
        self.is_running = False

class CLIPublisherNode(Node):
    def __init__(self):
        super().__init__('cli_publisher_node')
        self.publisher_ = self.create_publisher(PointStamped, 'raw_uwb_distances', 10)
        
        self.reader_a = SerialReaderThread(SERIAL_PORT_A, BAUD_RATE, DATA_REGEX, self.get_logger())
        self.reader_b = SerialReaderThread(SERIAL_PORT_B, BAUD_RATE, DATA_REGEX, self.get_logger())
        
        self.reader_a.start()
        self.reader_b.start()
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('UWB Publisher Node(CLI) Started.')

    def timer_callback(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='follower_uwb_center')
        point = Point(x=self.reader_a.latest_distance_m, y=self.reader_b.latest_distance_m, z=0.0)
        
        msg = PointStamped(header=header, point=point)
        self.publisher_.publish(msg)

    def on_shutdown(self):
        self.get_logger().info('UWB Publisher finishing...')
        self.reader_a.stop()
        self.reader_b.stop()
        # 스레드가 완전히 종료될 때까지 잠시 대기 (선택사항이지만 더 안정적)
        self.reader_a.join(timeout=1.0)
        self.reader_b.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = CLIPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 종료 시 필요한 정리 작업을 수행합니다.
        node.on_shutdown()
        node.destroy_node()
        # rclpy.shutdown()은 ros2 launch가 처리하므로 여기서 호출하지 않습니다.
        # 이 제거가 RCLError를 해결합니다.

if __name__ == '__main__':
    main()