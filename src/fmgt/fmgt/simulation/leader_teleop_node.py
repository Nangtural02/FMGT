# 파일명: leader_teleop_node.py
# 경로: fmgt/simulation/leader_teleop_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, os, math

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

# 리더의 이동 속도 (m/s)
LEADER_SPEED = 0.5

msg = """
가상 리더(보행자) 조종기 (8방향 이동)
---------------------------
움직임:
   q    w    e
   a    s    d
   z    x    c

w/s : 전/후진 (+/- X축)
a/d : 좌/우 이동 (+/- Y축)
q/e/z/c : 대각선 이동
x 또는 다른 키 : 정지

CTRL-C를 눌러 종료합니다.
"""

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class LeaderTeleopNode(Node):
    def __init__(self):
        super().__init__('leader_teleop_node')
        self.publisher_ = self.create_publisher(Twist, 'leader_teleop/cmd_vel', 10)
        self.settings = None
        if os.name != 'nt':
            self.settings = termios.tcgetattr(sys.stdin)

        self.timer = self.create_timer(0.1, self.publish_twist)
        self.key = ''
        
        print(msg)

    def publish_twist(self):
        self.key = get_key(self.settings)
        twist = Twist()
        
        # 대각선 이동 시 속도 보정 (속도 벡터의 크기를 LEADER_SPEED로 유지)
        diag_speed = LEADER_SPEED / math.sqrt(2)

        # ROS 좌표계 기준 (전방: +X, 좌측: +Y)
        if self.key == 'w':
            twist.linear.x = LEADER_SPEED
        elif self.key == 's':
            twist.linear.x = -LEADER_SPEED
        elif self.key == 'a':
            twist.linear.y = LEADER_SPEED
        elif self.key == 'd':
            twist.linear.y = -LEADER_SPEED
            
        # ★★★ 대각선 이동 로직 추가 ★★★
        elif self.key == 'q': # 전진 + 좌회전
            twist.linear.x = diag_speed
            twist.linear.y = diag_speed
        elif self.key == 'e': # 전진 + 우회전
            twist.linear.x = diag_speed
            twist.linear.y = -diag_speed
        elif self.key == 'z': # 후진 + 좌회전
            twist.linear.x = -diag_speed
            twist.linear.y = diag_speed
        elif self.key == 'c': # 후진 + 우회전
            twist.linear.x = -diag_speed
            twist.linear.y = -diag_speed

        # 'x' 키를 명시적인 정지 키로 추가
        elif self.key == 'x':
            pass # 모든 속도가 0.0인 Twist 메시지가 발행됨

        elif self.key == '\x03': # CTRL-C
            raise KeyboardInterrupt
        else:
            # w,a,s,d,q,e,z,c,x 가 아닌 다른 키를 눌러도 정지
            pass
        
        self.publisher_.publish(twist)

    def on_shutdown(self):
        if self.settings and os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        self.publisher_.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = LeaderTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCtrl+C 감지, 노드를 종료합니다.")
    finally:
        node.on_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()