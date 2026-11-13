<div align="center">

# **F1TENTH Reactive Control — Optimized Follow the Gap (Sakhir Circuit)**

**Unified long-horizon architecture featuring an adaptive safety bubble, disparity extension, and progressive braking in ROS 2 Humble**

**Author:** William Daniel Amaya Orellana  
**Institution:** Escuela Superior Politécnica del Litoral (ESPOL) — Mechatronics Engineering

</div>

## **1. Overview**

This repository contains the implementation and optimization of an autonomous reactive controller based on the **Follow the Gap** algorithm for **F1TENTH** 1:10 scale vehicles in the `f1tenth_gym_ros` simulator running on **ROS 2 Humble** (Ubuntu 22.04 LTS).

The algorithm is specifically tuned for the **Sakhir** circuit layout, resolving the classic trade-off between maximum straight-line speed and stability through tight hairpins via:

- **Long Planning Horizon (`11.0 m`) with Straight-Line Geometric Centering:** Projects the target point directly onto the midpoint of the free corridor and applies a steering deadband (`0.022 rad`) alongside directional damping (`alpha = 0.28`), completely eliminating high-speed weaving or oscillation (`12.0 m/s`).
- **Adaptive Metric Safety Bubble:** Dynamically adjusts the exclusion radius around the closest obstacle (`0.35 m` on clear straights to prevent lateral wall-bouncing, and `0.52 m` in corners to guarantee turning clearance).
- **Disparity Extension (`disparity_threshold = 0.45 m`):** Virtually expands inner corner edges based on the vehicle's half-width (`0.36 m`) to prevent apex clipping and collisions with inside walls.
- **Progressive Longitudinal Control:** Continuously modulates target speed based on forward clearance and Ackermann steering angle, incorporating acceleration and deceleration rate limiters to prevent wheel lockup or physics wall-tunneling in the simulator.
- **Live Telemetry & Stopwatch:** Real-time console tracking of current lap time, cumulative time, average lap time, and personal best across 10-lap endurance runs.

## **2. Step-by-Step Guide to Run the Simulation**

### **Step 1: Verify Prerequisites**
Ensure your system has **Ubuntu 22.04 LTS**, **ROS 2 Humble**, and the **`f1tenth_gym_ros`** simulator installed inside your ROS 2 workspace (`~/f1tenth_ws`).

### **Step 2: Clone and Build the Package**
1. Navigate to your workspace `src` folder and clone this repository:
```bash
cd ~/f1tenth_ws/src
git clone https://github.com/DanielReborn01/ReactiveControlsF1Tenth.git
```

2. Copy the Sakhir map files into your `f1tenth_gym_ros` directory:
```bash
cp ~/f1tenth_ws/src/ReactiveControlsF1Tenth/maps/Sakhir_map.* ~/f1tenth_ws/src/f1tenth_gym_ros/maps/
cp ~/f1tenth_ws/src/ReactiveControlsF1Tenth/maps/sim.yaml ~/f1tenth_ws/src/f1tenth_gym_ros/config/sim.yaml
```

3. Build the workspace using `colcon`:
```bash
cd ~/f1tenth_ws
colcon build --packages-select follow_the_gap_pkg f1tenth_gym_ros
source install/setup.bash
```

### **Step 3: Launch the Sakhir Simulator (Terminal 1)**
Open a terminal, source the workspace, and start the F1TENTH Gym ROS bridge:
```bash
cd ~/f1tenth_ws
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
Wait until the RViz2 window opens showing the **Sakhir** track and the ego vehicle spawned at the starting line.

### **Step 4: Run the Reactive Controller Node (Terminal 2)**
Open a second terminal, source the workspace, and start `gap_node`:
```bash
cd ~/f1tenth_ws
source install/setup.bash
ros2 run follow_the_gap_pkg gap_node
```
The vehicle will immediately accelerate down the main straight and display the live 10-lap stopwatch in Terminal 2.

### **Step 5: Resetting the Vehicle (Optional)**
If you want to restart the 10-lap run from the starting grid at any time, press `Ctrl + C` in both terminals, re-launch **Step 3** first, and then run **Step 4**.

## **3. Speed Profile & Control Parameters**

| Driving Regime | Trigger Condition | Target Speed | Key Parameters |
| :--- | :--- | :---: | :--- |
| **Clear Straight** | Front clearance > `13.5 m` and `abs(δ) ≤ 0.03 rad` | **`12.0 m/s`** | Bubble: `0.35 m`, Deadband: `0.022 rad`, `alpha = 0.28` |
| **Medium / Fast Corner** | Front clearance between `6.5 m` and `13.5 m` | **`7.0 m/s`** | Bubble: `0.52 m`, Early braking triggered at `13.5 m` |
| **Tight Hairpin (U-Turn)** | Front clearance < `6.5 m` or `abs(δ) > 0.24 rad` | **`2.2 m/s`** | Agile slew rate (`0.11 rad/scan`), full lock unlock `< 2.2 m` |

## **4. Node Architecture & Function Breakdown (`gap_node.py`)**

The ROS 2 node `ReactiveFollowGap` processes 2D LiDAR scans (`/scan`) and vehicle odometry (`/ego_racecar/odom`) at 40 Hz to publish Ackermann steering and velocity commands (`/drive`):

- **`__init__(self)`**: Initializes ROS 2 publishers and subscribers, defines geometric and dynamic parameters (horizon `11.0 m`, disparity threshold `0.45 m`, vehicle half-width `0.36 m`, speed profile `12.0 / 7.0 / 2.2 m/s`), and starts a `0.1 s` timer for the live console stopwatch.
- **`preprocess_lidar(self, ranges)`**: Replaces `inf` and `NaN` sensor readings with `max_lidar_dist` (`11.0 m`), applies a 7-sample moving average convolution filter to eliminate high-frequency LiDAR noise, and clips distances to the planning horizon.
- **`extend_disparities(self, ranges, angle_increment)`**: Scans adjacent LiDAR beams for sudden depth jumps exceeding `0.45 m` (wall edges or corner apexes) and projects the vehicle's half-width (`0.36 m`) across the deeper side using `arctan2(car_half_width, min_dist)` so the car clears inner walls safely.
- **`find_max_gap(self, free_space_ranges)`**: Masks all zeroed-out beams (safety bubble and out-of-range sectors) and returns the start and end indices of the longest contiguous navigable corridor.
- **`find_best_point(self, start_i, end_i, ranges, is_straight)`**: Selects the optimal steering target within the maximum gap. On straights (`is_straight = True`), it locks onto the exact geometric midpoint `(start_i + end_i) // 2` to prevent oscillation; in corners, it blends `40%` deepest beam and `60%` gap center to carve smooth racing lines.
- **`lidar_callback(self, data)`**: Executes the main reactive pipeline on every LiDAR scan: measures forward clearance (`±8°` cone), runs disparity extension, applies the adaptive metric safety bubble (`0.35 m` straight / `0.52 m` corner), computes smoothed steering with slew-rate limits, modulates target speed progressively, and publishes `AckermannDriveStamped` messages.
- **`odom_callback(self, msg)`**: Tracks vehicle coordinates `(x, y)` relative to the starting grid, detects lap completions once the car leaves the `8.0 m` start zone and re-enters within `2.0 m`, logs individual/cumulative/best lap times, and prints a summary report upon completing 10 collision-free laps.
- **`update_live_stopwatch(self)` & `format_time(self, seconds)`**: Formats elapsed seconds into `MM:SS.ms` and refreshes a single-line live telemetry readout in the terminal without cluttering lap logs.

## **5. Repository Structure**

```text
ReactiveControlsF1Tenth/
├── follow_the_gap_pkg/
│   ├── __init__.py
│   └── gap_node.py          # Main ROS 2 node 
├── maps/
│   ├── Sakhir_map.png       
│   ├── Sakhir_map.yaml      
│   └── sim.yaml             
├── resource/
├── test/
├── package.xml
├── setup.cfg
└── setup.py
```
