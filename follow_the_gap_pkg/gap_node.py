import sys
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')
        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/ego_racecar/odom'

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.window_size = 7
        self.max_lidar_dist = 11.0
        self.fov_ratio = 0.17
        self.disparity_threshold = 0.45
        self.car_half_width = 0.36

        self.straight_speed = 12.0
        self.corner_speed = 7.0
        self.hairpin_speed = 2.2
        self.brake_start_m = 13.5
        self.hairpin_dist_m = 6.5

        self.prev_speed = 2.5
        self.max_decel_step = 0.75
        self.max_accel_step = 0.45

        self.prev_steering = 0.0
        self.steer_deadband = 0.022

        self.target_laps = 10
        self.start_pos = None
        self.race_start_time = None
        self.lap_start_time = None
        self.lap_count = 0
        self.lap_times = []
        self.best_lap = float('inf')
        self.left_start_zone = False
        self.race_finished = False

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
        proc_ranges = np.array(ranges, dtype=np.float32)
        proc_ranges = np.nan_to_num(proc_ranges, posinf=self.max_lidar_dist, neginf=0.0)
        kernel = np.ones(self.window_size) / self.window_size
        proc_ranges = np.convolve(proc_ranges, kernel, mode='same')
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_lidar_dist)
        return proc_ranges

    def extend_disparities(self, ranges, angle_increment):
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
        masked = np.ma.masked_where(free_space_ranges == 0.0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked)
        if not slices:
            return 0, len(free_space_ranges) - 1
        max_slice = max(slices, key=lambda s: s.stop - s.start)
        return max_slice.start, max_slice.stop - 1

    def find_best_point(self, start_i, end_i, ranges, is_straight):
        gap_slice = ranges[start_i:end_i + 1]
        deepest_idx = np.argmax(gap_slice) + start_i
        center_idx = (start_i + end_i) // 2

        if is_straight:
            return center_idx
        return int(0.40 * deepest_idx + 0.60 * center_idx)

    def lidar_callback(self, data):
        raw_ranges = np.array(data.ranges, dtype=np.float32)
        raw_ranges = np.nan_to_num(raw_ranges, posinf=25.0, neginf=0.0)
        num_beams = len(raw_ranges)

        center_beam = num_beams // 2
        front_cone = raw_ranges[center_beam - 22:center_beam + 22]
        true_front_dist = float(np.percentile(front_cone, 12))
        is_straight = true_front_dist > 10.5

        fov_start = int(num_beams * self.fov_ratio)
        fov_end = int(num_beams * (1.0 - self.fov_ratio))
        proc_ranges = self.preprocess_lidar(raw_ranges[fov_start:fov_end])

        proc_ranges = self.extend_disparities(proc_ranges, data.angle_increment)

        closest_idx = np.argmin(proc_ranges)
        closest_dist = max(float(proc_ranges[closest_idx]), 0.1)
        bubble_m = 0.35 if is_straight else 0.52
        bubble_angle = np.arctan2(bubble_m, closest_dist)
        bubble_radius_beams = int(np.ceil(bubble_angle / data.angle_increment))

        min_idx = max(0, closest_idx - bubble_radius_beams)
        max_idx = min(len(proc_ranges) - 1, closest_idx + bubble_radius_beams)
        proc_ranges[min_idx:max_idx + 1] = 0.0

        gap_start, gap_end = self.find_max_gap(proc_ranges)
        best_idx = self.find_best_point(gap_start, gap_end, proc_ranges, is_straight=is_straight)

        actual_beam_idx = best_idx + fov_start
        raw_steering = data.angle_min + actual_beam_idx * data.angle_increment
        raw_steering = float(np.clip(raw_steering, -0.4189, 0.4189))

        if is_straight:
            if abs(raw_steering) < self.steer_deadband:
                raw_steering = 0.0
            else:
                raw_steering *= 0.45
            alpha = 0.28
            max_rate = 0.025
        else:
            alpha = 0.58
            max_rate = 0.4189 if true_front_dist < 2.2 else 0.11

        filtered_steering = alpha * raw_steering + (1.0 - alpha) * self.prev_steering
        steer_delta = np.clip(filtered_steering - self.prev_steering, -max_rate, max_rate)
        steering_angle = float(self.prev_steering + steer_delta)
        self.prev_steering = steering_angle

        if true_front_dist >= self.brake_start_m:
            dist_speed = self.straight_speed
        elif true_front_dist >= self.hairpin_dist_m:
            ratio = (true_front_dist - self.hairpin_dist_m) / (self.brake_start_m - self.hairpin_dist_m)
            dist_speed = self.corner_speed + (self.straight_speed - self.corner_speed) * ratio
        else:
            ratio = np.clip((true_front_dist - 2.3) / (self.hairpin_dist_m - 2.3), 0.0, 1.0)
            dist_speed = self.hairpin_speed + (self.corner_speed - self.hairpin_speed) * (ratio ** 1.3)

        abs_steer = abs(steering_angle)
        if abs_steer > 0.03:
            steer_ratio = np.clip((abs_steer - 0.03) / (0.14 - 0.03), 0.0, 1.0)
            steer_speed_cap = self.straight_speed - (self.straight_speed - self.corner_speed) * steer_ratio
            if abs_steer > 0.24:
                steer_speed_cap = min(steer_speed_cap, self.hairpin_speed)
            target_speed = min(dist_speed, steer_speed_cap)
        else:
            target_speed = dist_speed

        speed_diff = np.clip(target_speed - self.prev_speed, -self.max_decel_step, self.max_accel_step)
        speed = float(self.prev_speed + speed_diff)
        self.prev_speed = speed

        drive_msg = AckermannDriveStamped()
        drive_msg.drive = AckermannDrive()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
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
