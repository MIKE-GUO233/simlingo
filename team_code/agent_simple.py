"""Simple navigation agent that follows the provided route and avoids
basic obstacles using a front LiDAR sensor. The goal of this agent is to
provide a minimal example that mimics the interface of
``agent_simlingo.py`` so that the surrounding framework can call
``tick`` and ``run_step`` in the same way.

The agent:
* loads the global route supplied by the leaderboard framework,
* computes a target point using ``RoutePlanner``,
* applies a very small proportional controller to steer towards the
  target point,
* drives at a constant target speed (4 m/s) and uses a LiDAR based check
  to stop when an obstacle is detected in front of the ego vehicle.

This file intentionally keeps the implementation compact and does not
use any machine learning model. It is only meant as a light-weight
baseline that shows how to wire an agent into the benchmark.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import carla
import numpy as np
from leaderboard.autoagents import autonomous_agent

from team_code.nav_planner import RoutePlanner
import team_code.transfuser_utils as t_u


def get_entry_point() -> str:
    """Entry point for the evaluation framework."""
    return "SimpleAgent"


class SimpleAgent(autonomous_agent.AutonomousAgent):
    """Minimal agent that follows a route and avoids obstacles."""

    def setup(self, path_to_conf_file: str = "", route_index: int | None = None) -> None:
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.initialized = False

        # Route planner configuration (meters)
        self.route_planner_min_distance = 7.5
        self.route_planner_max_distance = 50.0

        # Navigation helpers
        self._route_planner: RoutePlanner | None = None
        self.lat_ref = 0.0
        self.lon_ref = 0.0

        # Vehicle control buffer
        self.control = carla.VehicleControl()

        # Target speed (m/s)
        self.target_speed = 4.0

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init(self) -> None:
        """Initialise the route planner once the global plan is available."""
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]
            earth_radius_equa = 6378137.0

            def equations(vars: tuple[float, float]) -> tuple[float, float]:
                x, y = vars
                eq1 = (
                    lon * math.cos(x * math.pi / 180.0)
                    - (locx * x * 180.0) / (math.pi * earth_radius_equa)
                    - math.cos(x * math.pi / 180.0) * y
                )
                eq2 = (
                    math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                    * earth_radius_equa
                    * math.cos(x * math.pi / 180.0)
                    + locy
                    - math.cos(x * math.pi / 180.0)
                    * earth_radius_equa
                    * math.log(math.tan((90.0 + x) * math.pi / 360.0))
                )
                return eq1, eq2

            # Simple numerical approximation using gradient descent
            step = 0.0001
            x, y = 0.0, 0.0
            for _ in range(1000):
                f1, f2 = equations((x, y))
                x -= f1 * step
                y -= f2 * step
            self.lat_ref, self.lon_ref = x, y
        except Exception:
            self.lat_ref, self.lon_ref = 0.0, 0.0

        self._route_planner = RoutePlanner(
            self.route_planner_min_distance,
            self.route_planner_max_distance,
            self.lat_ref,
            self.lon_ref,
        )
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    # ------------------------------------------------------------------
    # Sensor configuration
    # ------------------------------------------------------------------
    def sensors(self) -> list[Dict[str, Any]]:
        """Return the list of sensors required by this agent."""
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": -1.5,
                "y": 0.0,
                "z": 2.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_front",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "speed",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 0.0,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "rotation_frequency": 10,
                "points_per_second": 20000,
                "id": "lidar",
            },
        ]
        return sensors

    # ------------------------------------------------------------------
    # Per-step processing
    # ------------------------------------------------------------------
    def tick(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process sensor data and compute navigation targets."""
        gps = self._route_planner.convert_gps_to_carla(input_data["gps"][1])
        compass = t_u.preprocess_compass(input_data["imu"][1][-1])
        speed = input_data["speed"][1]["speed"]

        route = self._route_planner.run_step(np.append(gps, 0.0))
        if len(route) > 1:
            target, _ = route[1]
        else:
            target, _ = route[0]

        ego_target = t_u.inverse_conversion_2d(target[:2], gps[:2], compass)

        lidar = input_data.get("lidar")
        obstacle = False
        if lidar is not None:
            pts = lidar[1][:, :3]
            mask = (
                (pts[:, 0] > 0.0)
                & (pts[:, 0] < 8.0)
                & (np.abs(pts[:, 1]) < 1.0)
            )
            obstacle = bool(mask.any())

        return {"speed": speed, "target": ego_target, "obstacle": obstacle}

    # ------------------------------------------------------------------
    def run_step(self, input_data: Dict[str, Any], timestamp: float, sensors: Dict[str, Any] | None = None) -> carla.VehicleControl:
        """Compute the next control command."""
        self.step += 1
        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control

        data = self.tick(input_data)
        speed = data["speed"]
        target = data["target"]
        obstacle = data["obstacle"]

        # Simple steering: normalised angle to target point
        angle = math.atan2(target[1], target[0])
        steer = np.clip(angle / (math.pi / 2.0), -1.0, 1.0)

        # Speed control
        throttle = 0.5 if speed < self.target_speed else 0.0
        brake = 0.0
        if obstacle:
            throttle = 0.0
            brake = 1.0

        self.control.steer = float(steer)
        self.control.throttle = float(throttle)
        self.control.brake = float(brake)
        return self.control

    # ------------------------------------------------------------------
    def destroy(self, results: Dict[str, Any] | None = None) -> None:  # pylint: disable=unused-argument
        """Called after a route finished."""
        self._route_planner = None
        self.initialized = False

