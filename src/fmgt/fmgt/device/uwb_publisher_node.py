# fmgt/uwb_publisher_node.py
import rclpy
import serial # pyright: ignore[reportMissingModuleSource]
import re
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header

SERIAL_PORT_A = '/dev/ttyACM0' #left(blue)
SERIAL_PORT_B = '/dev/ttyACM1' #right(red)
BAUD_RATE = 115200

CLI_DATA_REGEX = re.compile(r"distance\[cm\]=(\d+)")
QANI_DATA_REGEX = re.compile(r'"D_cm":(\d+)')

class UWBPublisherNode(Node):
    def __init__(self):
        super().__init__('uwb_publisher_node')
        self.publisher_ = self.create_publisher(PointStamped, 'raw_uwb_distances', 10)

        self.declare_parameter('firmware_build', "QANI")
        self.firmware_build = self.get_parameter('firmware_build').value
        if self.firmware_build == "QANI":
            data_regex = QANI_DATA_REGEX
        elif self.firmware_build == "CLI":
            data_regex = CLI_DATA_REGEX
        else: 
            raise Exception("incompatiable firmware_build")
        
        self.reader_a = SerialReaderThread(SERIAL_PORT_A, BAUD_RATE, data_regex, self.get_logger())
        self.reader_b = SerialReaderThread(SERIAL_PORT_B, BAUD_RATE, data_regex, self.get_logger())
        
        self.reader_a.start()
        self.reader_b.start()
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f'UWB Publisher Node({self.firmware_build} Build) Started.')

    def timer_callback(self):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='follower_uwb_center')
        point = Point(x=self.reader_a.latest_distance_m, y=self.reader_b.latest_distance_m, z=0.0)
        
        msg = PointStamped(header=header, point=point)
        self.publisher_.publish(msg)

    def on_shutdown(self):
        self.get_logger().info('UWB Publisher finishing...')
        self.reader_a.stop()
        self.reader_b.stop()
        self.reader_a.join(timeout=1.0)
        self.reader_b.join(timeout=1.0)

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
        try:
            while self.is_running and rclpy.ok():
                try:
                    if self.ser is None:
                        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                        self._logger.info(f"Connection with {self.port} Success.")

                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        match = self.regex.search(line)
                        if match:
                            self.latest_distance_m = int(match.group(1)) / 100.0
                
                except serial.SerialException:
                    if self.ser:
                        self.ser.close()
                    self.ser = None
                    if self.is_running:
                        self._logger.warn(f"{self.port} Disconnected. Retry after 2 second...")
                        time.sleep(2)
                
                except (ValueError, AttributeError) as e:
                    self._logger.warn(f"Data parsing error in {self.port} : {e}")
                    continue
        
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self._logger.info(f"Connection in {self.port} was successfully finished")

    def stop(self):
        self.is_running = False

def main(args=None):
    rclpy.init(args=args)
    node = UWBPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
if __name__ == '__main__':
    main()