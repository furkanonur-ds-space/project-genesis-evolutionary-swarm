from __future__ import annotations

import math

import numpy as np
import pygame

from brain import NeuralNetwork
from food import Food, FoodManager


class Agent:
    def __init__(
        self,
        position: pygame.Vector2,
        brain: NeuralNetwork | None = None,
        max_speed: float = 3.0,
        max_force: float = 0.22,
    ) -> None:
        self.position = pygame.Vector2(position)

        heading = pygame.Vector2(1, 0).rotate(np.random.uniform(0, 360))
        self.velocity = heading * np.random.uniform(0.5, max_speed)
        self.acceleration = pygame.Vector2()

        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn_radians = 0.55

        self.starting_health = 100.0
        self.max_health = 150.0
        self.health = self.starting_health
        self.health_decay = 0.5
        self.alive = True

        self.radius = 7.0
        self.eat_radius = 10.0

        self.age_frames = 0
        self.food_eaten = 0
        self.brain = brain or NeuralNetwork()

    @property
    def fitness(self) -> float:
        return float(self.age_frames + self.food_eaten * 50)

    def _heading(self) -> pygame.Vector2:
        if self.velocity.length_squared() < 1e-9:
            return pygame.Vector2(1, 0)
        return self.velocity.normalize()

    def _sensor_inputs(self, food_manager: FoodManager, world_width: int, world_height: int) -> tuple[np.ndarray, Food | None, float]:
        nearest_food, distance, offset = food_manager.nearest(self.position)
        max_distance = math.hypot(world_width, world_height)

        if nearest_food is None:
            normalized_distance = 1.0
            angle_to_food = 0.0
        else:
            normalized_distance = min(distance / max_distance, 1.0)
            heading = self._heading()
            target = offset.normalize() if offset.length_squared() > 0 else heading
            dot = float(np.clip(heading.dot(target), -1.0, 1.0))
            cross = heading.x * target.y - heading.y * target.x
            signed_angle = math.atan2(cross, dot)
            angle_to_food = signed_angle / math.pi

        health_level = float(np.clip(self.health / self.starting_health, 0.0, 1.0))

        return np.array([normalized_distance, angle_to_food, health_level], dtype=np.float32), nearest_food, distance

    def _apply_controls(self, thrust: float, steering: float) -> None:
        heading = self._heading()
        desired_heading = heading.rotate_rad(steering * self.max_turn_radians)
        desired_velocity = desired_heading * (max(0.0, thrust) * self.max_speed)

        steering_force = desired_velocity - self.velocity
        if steering_force.length() > self.max_force:
            steering_force.scale_to_length(self.max_force)

        self.acceleration += steering_force

    def _integrate(self, world_width: int, world_height: int) -> None:
        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        self.position += self.velocity
        self.position.x %= world_width
        self.position.y %= world_height
        self.acceleration.update(0.0, 0.0)

    def update(self, food_manager: FoodManager, world_width: int, world_height: int) -> None:
        if not self.alive:
            return

        self.age_frames += 1
        self.health -= self.health_decay

        sensors, nearest_food, nearest_distance = self._sensor_inputs(food_manager, world_width, world_height)
        output = self.brain.forward(sensors)

        thrust = float((output[0] + 1.0) * 0.5)
        steering = float(np.clip(output[1], -1.0, 1.0))

        self._apply_controls(thrust, steering)
        self._integrate(world_width, world_height)

        if nearest_food is not None and nearest_distance <= self.eat_radius:
            food_manager.consume(nearest_food)
            self.health = min(self.max_health, self.health + 20.0)
            self.food_eaten += 1

        if self.health <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        heading = self._heading()
        left = heading.rotate(140)
        right = heading.rotate(-140)

        nose = self.position + heading * (self.radius * 1.7)
        rear_left = self.position + left * self.radius
        rear_right = self.position + right * self.radius

        health_ratio = float(np.clip(self.health / self.max_health, 0.0, 1.0))
        color = (int(255 * (1.0 - health_ratio)), int(255 * health_ratio), 50)

        pygame.draw.polygon(surface, color, [nose, rear_left, rear_right])
