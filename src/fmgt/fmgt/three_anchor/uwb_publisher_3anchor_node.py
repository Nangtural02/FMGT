# 파일명: uwb_publisher_node.py
"""
UWB Publisher Node (3-Anchor)

3개의 UWB 앵커(전방 좌/우, 후방 중앙)로부터 시리얼 통신을 통해 거리 데이터를
수신하고, 이를 하나의 ROS 2 메시지로 묶어 발행합니다. PointStamped 메시지의
x, y, z 필드를 각각 d_a, d_b, d_c 거리 값으로 사용합니다.

- 시리얼 포트 정의:
  - SERIAL_PORT_A: /dev/ttyACM0 (전방 좌측 앵커)
  - SERIAL_PORT_B: /dev/ttyACM1 (전방 우측 앵커)
  - SERIAL_PORT_C: /dev/ttyACM2 (후방 중앙 앵커)

- 발행 토픽:
  - raw_uwb_distances (geometry_msgs/PointStamped): 3개의 거리 측정치를 담은 메시지.
    (point.x: d_a, point.y: d_b, point.z: d_c)
"""
import rclpy
import serial
import re
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header

SERIAL_PORT_A = '/dev/ttyACM0'
SERIAL_PORT_B = '/dev/ttyACM1'
SERIAL_PORT_C = '/dev/ttyACM2'
BAUD_RATE = 115200

CLI_DATA_REGEX = re.compile(r"distance\[cm\]=(\d+)")
QANI_DATA_REGEX = re.compile(r'"D_cm":(\d+)')

class UWBPublisherNode(Node):
    def __init__(self):
        super().__init__('uwb_publisher_node')
        self.publisher_ = self.create_publisher(PointStamped, 'raw_uwb_distances', 10)

        self.declare_parameter('firmware_build', "QANI")
        self.firmware_build = self.get_parameter('firmware_build').value
        data_regex = QANI_DATA_REGEX if self.firmware_build == "QANI" else CLI_DATA_REGEX
        
        self.reader_a = SerialReaderThread(SERIAL_PORT_A, BAUD_RATE, data_regex, self.get_logger())
        self.reader_b = SerialReaderThread(SERIAL_PORT_B, BAUD_RATE, data_regex, self.get_logger())
        self.reader_c = SerialReaderThread(SERIAL_PORT_C, BAUD_RATE, data_regex, self.get_logger())
        
        self.reader_a.start()
        self.reader_b.start()
        self.reader_c.start()
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f'UWB Publisher Node (3-Anchor, {self.firmware_build} Build) Started.')

    def timer_callback(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='follower_uwb_center')
        point = Point(x=self.reader_a.latest_distance_m, 
                      y=self.reader_b.latest_distance_m, 
                      z=self.reader_c.latest_distance_m)
        msg = PointStamped(header=header, point=point)
        self.publisher_.publish(msg)

    def on_shutdown(self):
        self.get_logger().info('UWB Publisher finishing...')
        self.reader_a.stop(); self.reader_b.stop(); self.reader_c.stop()
        self.reader_a.join(timeout=1.0); self.reader_b.join(timeout=1.0); self.reader_c.join(timeout=1.0)

class SerialReaderThread(threading.Thread):
    def __init__(self, port, baudrate, regex, logger):
        super().__init__(daemon=True)
        self.port, self.baudrate, self.regex, self._logger = port, baudrate, regex, logger
        self.latest_distance_m, self.is_running, self.ser = 0.0, True, None

    def run(self):
        try:
            while self.is_running and rclpy.ok():
                try:
                    if self.ser is None:
                        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                        self._logger.info(f"Connection with {self.port} Success.")
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        match = self.regex.search(line)
                        if match: self.latest_distance_m = int(match.group(1)) / 100.0
                except serial.SerialException:
                    if self.ser: self.ser.close()
                    self.ser = None
                    if self.is_running:
                        self._logger.warn(f"{self.port} Disconnected. Retry after 2 second...")
                        time.sleep(2)
                except (ValueError, AttributeError) as e:
                    self._logger.warn(f"Data parsing error in {self.port} : {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self._logger.info(f"Connection in {self.port} was successfully finished")

    def stop(self): self.is_running = False

def main(args=None):
    rclpy.init(args=args)
    node = UWBPublisherNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()