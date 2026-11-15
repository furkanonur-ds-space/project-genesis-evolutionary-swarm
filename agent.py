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

        self._base_max_speed = max_speed
        self.max_force = max_force
        self.max_turn_radians = 0.55

        self.starting_health = 100.0
        self.max_health = 200.0
        self.health = self.starting_health
        self.health_decay = 0.4
        self.alive = True

        self.radius = 6.0           # Başlangıç boyutu
        self.eat_radius = 10.0

        self.age_frames = 0
        self.food_eaten = 0
        self.agents_eaten = 0       # Kaç ajan yedi
        self.brain = brain or NeuralNetwork()

        # Büyüme sistemi: yem yedikçe büyür
        self.growth = 1.0           # 1.0 = normal, 2.0 = 2x büyük

    @property
    def fitness(self) -> float:
        """Son hayatta kalan = en yüksek fitness.
        Yem yemek (ufak ödül) + ajan yemek (dev ödül) + uzun yaşamak."""
        return float(
            self.age_frames
            + self.food_eaten * 10       # Yem yemek artık daha az değerli
            + self.agents_eaten * 1000   # Ajan yemek inanılmaz değerli
        )

    @property
    def actual_radius(self) -> float:
        """Görsel ve çarpışma yarıçapı — büyüdükçe artar."""
        return self.radius * self.growth

    @property
    def max_speed(self) -> float:
        """Büyüdükçe biraz hızlanır (avantaj)."""
        return self._base_max_speed * (1.0 + (self.growth - 1.0) * 0.3)

    def _heading(self) -> pygame.Vector2:
        if self.velocity.length_squared() < 1e-9:
            return pygame.Vector2(1, 0)
        return self.velocity.normalize()

    def _sensor_inputs(
        self,
        food_manager: FoodManager,
        agents: list[Agent],
        world_width: int,
        world_height: int,
    ) -> tuple[np.ndarray, Food | None, float]:
        # --- Yemek Sensörü ---
        nearest_food, f_dist, f_offset = food_manager.nearest(self.position)
        max_distance = math.hypot(world_width, world_height)

        if nearest_food is None:
            norm_f_dist = 1.0
            angle_to_food = 0.0
        else:
            norm_f_dist = min(f_dist / max_distance, 1.0)
            heading = self._heading()
            target = f_offset.normalize() if f_offset.length_squared() > 0 else heading
            dot = float(np.clip(heading.dot(target), -1.0, 1.0))
            cross = heading.x * target.y - heading.y * target.x
            angle_to_food = math.atan2(cross, dot) / math.pi

        # --- Ajan Sensörü (Göz) ---
        nearest_agent = None
        min_a_dist_sq = float('inf')
        a_offset = pygame.Vector2()

        for other in agents:
            if not other.alive or other is self:
                continue
            offset = other.position - self.position
            dist_sq = offset.length_squared()
            if dist_sq < min_a_dist_sq:
                min_a_dist_sq = dist_sq
                nearest_agent = other
                a_offset = offset

        if nearest_agent is None:
            norm_a_dist = 1.0
            angle_to_agent = 0.0
            relative_size = 0.0  # Ortada kimse yok
        else:
            a_dist = math.sqrt(min_a_dist_sq)
            norm_a_dist = min(a_dist / max_distance, 1.0)
            heading = self._heading()
            target = a_offset.normalize() if a_offset.length_squared() > 0 else heading
            dot = float(np.clip(heading.dot(target), -1.0, 1.0))
            cross = heading.x * target.y - heading.y * target.x
            angle_to_agent = math.atan2(cross, dot) / math.pi
            
            # Boyut farkı (Tehlike/Av sensörü)
            # Pozitif = ben ondan büyüğüm (ye)
            # Negatif = o benden büyük (kaç)
            relative_size = float(np.clip(self.growth - nearest_agent.growth, -1.0, 1.0))

        health_level = float(np.clip(self.health / self.starting_health, 0.0, 1.0))

        sensors = np.array([
            norm_f_dist, angle_to_food, health_level,
            norm_a_dist, angle_to_agent, relative_size
        ], dtype=np.float32)

        return sensors, nearest_food, f_dist

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

    def grow(self, amount: float = 0.08) -> None:
        """Yem yedikçe büyü. Max 3x orijinal boyut."""
        self.growth = min(3.0, self.growth + amount)

    def can_eat(self, other: Agent) -> bool:
        """Bu ajan, diğerini yiyebilir mi? Sadece daha büyükse."""
        if not other.alive or not self.alive:
            return False
        # En az %20 daha büyük olmalı
        return self.growth > other.growth * 1.2

    def update(
        self,
        food_manager: FoodManager,
        agents: list[Agent],
        world_width: int,
        world_height: int,
    ) -> None:
        if not self.alive:
            return

        self.age_frames += 1
        # Büyüdükçe canın azalması yavaşlar (Avcı olmanın ödülü)
        # 3x büyüklükte bir ajan, normalin çok daha azı kadar can kaybeder
        actual_decay = self.health_decay / (1.0 + (self.growth - 1.0) * 2.0)
        self.health -= actual_decay

        sensors, nearest_food, nearest_distance = self._sensor_inputs(
            food_manager, agents, world_width, world_height
        )
        output = self.brain.forward(sensors)

        thrust = float((output[0] + 1.0) * 0.5)
        steering = float(np.clip(output[1], -1.0, 1.0))

        self._apply_controls(thrust, steering)
        self._integrate(world_width, world_height)

        # Yem yeme
        actual_eat_r = self.eat_radius * self.growth
        if nearest_food is not None and nearest_distance <= actual_eat_r:
            food_manager.consume(nearest_food)
            self.health = min(self.max_health, self.health + 20.0)
            self.food_eaten += 1
            self.grow(0.08)  # Her yem = biraz büyüme

        if self.health <= 0.0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        heading = self._heading()
        ar = self.actual_radius

        left = heading.rotate(140)
        right = heading.rotate(-140)

        nose = self.position + heading * (ar * 1.7)
        rear_left = self.position + left * ar
        rear_right = self.position + right * ar

        health_ratio = float(np.clip(self.health / self.max_health, 0.0, 1.0))
        # Büyükler kırmızıya kayar (tehlikeli görünüm)
        if self.growth > 1.5:
            r = int(200 + 55 * min(1.0, (self.growth - 1.5)))
            g = int(80 * health_ratio)
            b = int(30)
        else:
            r = int(255 * (1.0 - health_ratio))
            g = int(255 * health_ratio)
            b = 50
        color = (min(255, r), min(255, g), min(255, b))

        pygame.draw.polygon(surface, color, [nose, rear_left, rear_right])
