from __future__ import annotations

import numpy as np
import pygame

from agent import Agent
from food import FoodManager
from genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 760
FPS = 60

POPULATION_SIZE = 50
FOOD_COUNT = 30
GENERATION_FRAME_LIMIT = 1800

BACKGROUND_COLOR = (18, 20, 24)
TEXT_COLOR = (230, 230, 230)


def create_agents(pop_size: int, world_width: int, world_height: int, brains=None) -> list[Agent]:
    agents: list[Agent] = []
    for i in range(pop_size):
        position = pygame.Vector2(
            np.random.uniform(0, world_width),
            np.random.uniform(0, world_height),
        )
        brain = None if brains is None else brains[i]
        agents.append(Agent(position=position, brain=brain))
    return agents


def draw_ui(
    surface: pygame.Surface,
    font: pygame.font.Font,
    generation: int,
    alive_count: int,
    best_fitness: float,
    frame_count: int,
) -> None:
    lines = [
        f"Generation: {generation}",
        f"Alive: {alive_count}",
        f"Best Fitness: {best_fitness:.1f}",
        f"Frame: {frame_count}/{GENERATION_FRAME_LIMIT}",
    ]

    for idx, text in enumerate(lines):
        label = font.render(text, True, TEXT_COLOR)
        surface.blit(label, (16, 14 + idx * 26))


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Project Genesis - Evolutionary Swarm")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)

    ga = GeneticAlgorithm(
        GeneticAlgorithmConfig(
            population_size=POPULATION_SIZE,
            elite_fraction=0.2,
            mutation_rate=0.05,
            mutation_strength=0.12,
        )
    )

    generation = 1
    generation_frame = 0
    best_fitness = 0.0

    agents = create_agents(POPULATION_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT)
    food_manager = FoodManager(WINDOW_WIDTH, WINDOW_HEIGHT, FOOD_COUNT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        generation_frame += 1

        for agent in agents:
            agent.update(food_manager, WINDOW_WIDTH, WINDOW_HEIGHT)

        alive_count = sum(1 for agent in agents if agent.alive)

        if alive_count == 0 or generation_frame >= GENERATION_FRAME_LIMIT:
            next_brains = ga.create_next_generation(agents)
            best_fitness = max(best_fitness, ga.last_generation_best)

            generation += 1
            generation_frame = 0
            agents = create_agents(POPULATION_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, brains=next_brains)
            food_manager.reset(FOOD_COUNT)

        screen.fill(BACKGROUND_COLOR)
        food_manager.draw(screen)
        for agent in agents:
            agent.draw(screen)

        draw_ui(screen, font, generation, alive_count, best_fitness, generation_frame)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run()
