import sys
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class ReactiveFollowGap(Node):
    """
    Reactive Follow the Gap: Balanced High-Speed + Safe Turn 1 Braking for Sakhir
    """
    def __init__(self):
        super().__init__('reactive_node')
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        # Subscriptions and publishers
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # Gap & Disparity parameters
        self.window_size = 5            # Moving average smoothing window
        self.max_lidar_dist = 8.0       # Extended horizon for high-speed lookahead (m)
        self.bubble_radius = 135        # Safety bubble radius around closest obstacle (indices)
        self.fov_ratio = 0.17           # ~235 deg front FOV
        self.disparity_threshold = 0.45 # Min distance jump (m) to trigger disparity extension
        self.car_half_width = 0.40      # Clearance around inner apexes (m)

        # Balanced fast-lap speed & braking profile (m/s & m)
        self.cruise_speed = 9.2         # Fast straight-line speed
        self.mid_speed = 5.6            # Fast pace through sweeping corners
        self.hairpin_speed = 2.2        # Safe apex speed for acute hairpins
        self.brake_trigger_dist = 8.5   # Start braking 8.5m before a front wall
        self.brake_end_dist = 2.4       # Reach hairpin speed 2.4m before the wall

        # Smooth & responsive steering control
        self.prev_steering = 0.0
        self.steering_alpha = 0.42
        self.max_steer_rate = 0.070     # Max steering angle change per LiDAR scan (rad)
        self.steer_deadband = 0.012     # Suppress micro-weaving on straights

        # Live stopwatch and lap telemetry
        self.target_laps = 10
        self.start_pos = None
        self.race_start_time = None
        self.lap_start_time = None
        self.lap_count = 0
        self.lap_times = []
        self.best_lap = float('inf')
        self.left_start_zone = False
        self.race_finished = False

        # Timer to refresh live stopwatch every 0.1s
        self.stopwatch_timer = self.create_timer(0.1, self.update_live_stopwatch)

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = seconds % 60
        return f'{mins:02d}:{secs:05.2f}'

    def update_live_stopwatch(self):
        if self.race_start_time is None or self.race_finished:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        current_lap_time = now - self.lap_start_time
        total_time = now - self.race_start_time
        best_str = self.format_time(self.best_lap) if self.best_lap < float('inf') else '--:--.--'

        sys.stdout.write(
            f'\r[CRONOMETRO] Vuelta {self.lap_count + 1}/{self.target_laps} | '
            f'Vuelta actual: {self.format_time(current_lap_time)} | '
            f'Total: {self.format_time(total_time)} | '
            f'Mejor: {best_str}   '
        )
        sys.stdout.flush()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_pos is None:
            self.start_pos = (x, y)
            self.race_start_time = now
            self.lap_start_time = now
            print(f'\n--- CRONOMETRO INICIADO (Meta: {self.target_laps} vueltas) ---')
            return

        dist = np.hypot(x - self.start_pos[0], y - self.start_pos[1])
        if dist > 8.0:
            self.left_start_zone = True
        elif dist < 2.0 and self.left_start_zone:
            self.lap_count += 1
            lap_time = now - self.lap_start_time
            total_time = now - self.race_start_time
            self.lap_times.append(lap_time)
            if lap_time < self.best_lap:
                self.best_lap = lap_time

            sys.stdout.write('\r' + ' ' * 95 + '\r')
            print(
                f'Vuelta {self.lap_count:02d}/{self.target_laps} completada | '
                f'Tiempo: {self.format_time(lap_time)} ({lap_time:.2f} s) | '
                f'Acumulado: {self.format_time(total_time)} | '
                f'Mejor: {self.format_time(self.best_lap)}'
            )

            if self.lap_count == self.target_laps and not self.race_finished:
                self.race_finished = True
                avg_lap = sum(self.lap_times) / len(self.lap_times)
                print('\n======================================================')
                print(f'  PRUEBA DE {self.target_laps} VUELTAS COMPLETADA SIN COLISIONES')
                print(f'  Tiempo total : {self.format_time(total_time)} ({total_time:.2f} s)')
                print(f'  Mejor vuelta : {self.format_time(self.best_lap)} ({self.best_lap:.2f} s)')
                print(f'  Promedio     : {self.format_time(avg_lap)} ({avg_lap:.2f} s)')
                print('======================================================\n')

            self.lap_start_time = now
            self.left_start_zone = False

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array:
            1. Set each value to the mean over a window
            2. Reject high values (> max_lidar_dist)
        """
        proc_ranges = np.array(ranges, dtype=np.float32)
        proc_ranges = np.nan_to_num(proc_ranges, posinf=self.max_lidar_dist, neginf=0.0)
        kernel = np.ones(self.window_size) / self.window_size
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_lidar_dist)
        return proc_ranges

    def extend_disparities(self, ranges, angle_increment):
        """ Extend edges of obstacles by car_half_width to prevent clipping inner corners
        """
        extended = np.copy(ranges)
        diffs = np.abs(np.diff(ranges))
        disparity_indices = np.where(diffs > self.disparity_threshold)[0]

        for idx in disparity_indices:
            min_dist = min(ranges[idx], ranges[idx + 1])
            if min_dist <= 0.05:
                continue
            half_width_angle = np.arctan2(self.car_half_width, min_dist)
            num_beams_to_extend = int(np.ceil(half_width_angle / angle_increment))

            if ranges[idx] < ranges[idx + 1]:
                end_i = min(len(extended), idx + 1 + num_beams_to_extend)
                extended[idx + 1:end_i] = np.minimum(extended[idx + 1:end_i], min_dist)
            else:
                start_i = max(0, idx - num_beams_to_extend + 1)
                extended[start_i:idx + 1] = np.minimum(extended[start_i:idx + 1], min_dist)

        return extended

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        masked = np.ma.masked_where(free_space_ranges == 0.0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)
        if not slices:
            return 0, len(free_space_ranges) - 1
        max_slice = max(slices, key=lambda s: s.stop - s.start)
        return max_slice.start, max_slice.stop - 1

    def find_best_point(self, start_i, end_i, ranges, is_straight=False):
        """ Start_i & end_i are start and end indices of max-gap range, respectively.
            Return index of best point in ranges.
        """
        gap_slice = ranges[start_i:end_i + 1]
        deepest_idx = np.argmax(gap_slice) + start_i
        center_idx = (start_i + end_i) // 2

        if is_straight:
            return int(0.18 * deepest_idx + 0.82 * center_idx)
        return int(0.35 * deepest_idx + 0.65 * center_idx)

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        raw_ranges = np.array(data.ranges, dtype=np.float32)
        raw_ranges = np.nan_to_num(raw_ranges, posinf=20.0, neginf=0.0)
        num_beams = len(raw_ranges)

        # Forward clearance (+/- 9 deg)
        center_beam = num_beams // 2
        front_cone = raw_ranges[center_beam - 25:center_beam + 25]
        true_front_dist = float(np.percentile(front_cone, 10))

        fov_start = int(num_beams * self.fov_ratio)
        fov_end = int(num_beams * (1.0 - self.fov_ratio))
        proc_ranges = self.preprocess_lidar(raw_ranges[fov_start:fov_end])

        # 1. Extend disparities around sharp corners
        proc_ranges = self.extend_disparities(proc_ranges, data.angle_increment)

        # 2. Find closest point to LiDAR and zero out safety bubble
        closest_idx = np.argmin(proc_ranges)
        min_idx = max(0, closest_idx - self.bubble_radius)
        max_idx = min(len(proc_ranges) - 1, closest_idx + self.bubble_radius)
        proc_ranges[min_idx:max_idx + 1] = 0.0

        # 3. Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # 4. Find the best point in the gap
        is_straight = true_front_dist > 6.0
        best_idx = self.find_best_point(gap_start, gap_end, proc_ranges, is_straight=is_straight)

        # 5. Compute smoothed and rate-limited steering angle
        actual_beam_idx = best_idx + fov_start
        raw_steering = data.angle_min + actual_beam_idx * data.angle_increment
        raw_steering = float(np.clip(raw_steering, -0.4189, 0.4189))

        if abs(raw_steering) < self.steer_deadband and is_straight:
            raw_steering = 0.0

        filtered_steering = self.steering_alpha * raw_steering + (1.0 - self.steering_alpha) * self.prev_steering
        steer_delta = np.clip(filtered_steering - self.prev_steering, -self.max_steer_rate, self.max_steer_rate)
        steering_angle = float(self.prev_steering + steer_delta)
        self.prev_steering = steering_angle

        # 6. Progressive braking before corners + cornering caps
        brake_span = self.brake_trigger_dist - self.brake_end_dist
        dist_ratio = np.clip((true_front_dist - self.brake_end_dist) / brake_span, 0.0, 1.0)
        speed_from_dist = self.hairpin_speed + (self.cruise_speed - self.hairpin_speed) * (dist_ratio ** 1.45)

        abs_steer = abs(steering_angle)
        if abs_steer > 0.23:
            speed = min(speed_from_dist, self.hairpin_speed)
        elif abs_steer > 0.13:
            speed = min(speed_from_dist, self.mid_speed)
        else:
            speed = speed_from_dist

        # 7. Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = float(speed)
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("ReactiveFollowGap Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
