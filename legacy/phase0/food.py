from __future__ import annotations

from dataclasses import dataclass
import random

import pygame


@dataclass
class Food:
    position: pygame.Vector2


class FoodManager:
    def __init__(self, width: int, height: int, initial_count: int) -> None:
        self.width = width
        self.height = height
        self.food_items: list[Food] = []
        self.spawn_many(initial_count)

    def _random_position(self) -> pygame.Vector2:
        return pygame.Vector2(
            random.uniform(0.0, float(self.width - 1)),
            random.uniform(0.0, float(self.height - 1)),
        )

    def spawn_one(self) -> None:
        self.food_items.append(Food(position=self._random_position()))

    def spawn_many(self, count: int) -> None:
        for _ in range(count):
            self.spawn_one()

    def consume(self, food: Food) -> None:
        try:
            self.food_items.remove(food)
        except ValueError:
            return
        self.spawn_one()

    def nearest(self, position: pygame.Vector2) -> tuple[Food | None, float, pygame.Vector2]:
        if not self.food_items:
            return None, float("inf"), pygame.Vector2()

        nearest_food: Food | None = None
        nearest_distance = float("inf")
        nearest_offset = pygame.Vector2()

        for food in self.food_items:
            offset = food.position - position
            distance = offset.length()
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_food = food
                nearest_offset = offset

        return nearest_food, nearest_distance, nearest_offset

    def reset(self, count: int) -> None:
        self.food_items.clear()
        self.spawn_many(count)

    def draw(self, surface: pygame.Surface) -> None:
        for food in self.food_items:
            pygame.draw.circle(surface, (60, 220, 70), food.position, 4)

    def __len__(self) -> int:
        return len(self.food_items)
