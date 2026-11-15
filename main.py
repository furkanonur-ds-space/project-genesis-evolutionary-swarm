"""
Project Genesis — "The Egg"
Faz 0: Hayatta Kalma

Mantık:
  1. SharedBrain oluşturulur (tek beyin)
  2. Her nesilde, SharedBrain'den 50 ajan kopyalanır (hafif mutasyonla)
  3. Ajanlar dünyada yem toplar, hayatta kalır
  4. Nesil sonunda en iyi ajanlar SharedBrain'i günceller
  5. Beyin nesiller arası "olgunlaşır"

  Eski sistemden farkı:
  - Eskiden: Her ajan bağımsız beyin, GA ile evrim
  - Şimdi: Tek SharedBrain, tüm ajanlar onun kopyası
  - "The Egg" felsefesi: Her hayat, tek bir ruhu besliyor
"""

from __future__ import annotations

import numpy as np
import pygame

from agent import Agent
from brain import NeuralNetwork
from food import FoodManager
from shared_brain import SharedBrain

# ---------------------------------------------------------------------------
# World / timing
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 760
FPS = 60

POPULATION_SIZE = 50
FOOD_COUNT = 30
GENERATION_FRAME_LIMIT = 1800

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BACKGROUND_COLOR = (18, 20, 24)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (100, 220, 180)
DIM_COLOR = (140, 140, 140)
SOUL_COLOR = (255, 200, 80)          # SharedBrain "ruh" rengi

# ---------------------------------------------------------------------------
# Maturity colors — beyin büyüdükçe renk değişir
# ---------------------------------------------------------------------------
MATURITY_COLORS = {
    "Yenidoğan": (200, 200, 200),
    "Bebek":     (180, 220, 255),
    "Çocuk":     (100, 220, 180),
    "Genç":      (255, 200, 80),
    "Olgun":     (255, 120, 200),
}


def create_agents_from_shared_brain(
    shared_brain: SharedBrain,
    pop_size: int,
    world_width: int,
    world_height: int,
    rng: np.random.Generator,
    prev_elites: list[NeuralNetwork] | None = None,
) -> list[Agent]:
    """SharedBrain'den çeşitli yöntemlerle ajan oluştur.
    
    Dağılım:
    - %15 sadık kopyalar (SharedBrain'i koru)
    - %25 mutasyonlu kopyalar (yakın keşif)
    - %20 elite crossover (önceki neslin en iyilerini karıştır)
    - %20 tamamen rastgele beyin (yeni ruhlar — sıfırdan dene)
    - %20 ağır mutasyon (çılgın denemeler)
    
    Neden rastgele beyinler var?
    Eğer SharedBrain kötü bir alışkanlık edindiyse (dönme gibi),
    tüm kopyaları da aynı hatayı yapar. Rastgele beyinler bu
    "kötü döngüyü" kırar — birisi tesadüfen iyi bir şey yaparsa,
    SharedBrain onu öğrenir.
    """
    agents: list[Agent] = []
    
    def _make_agent(brain: NeuralNetwork) -> Agent:
        pos = pygame.Vector2(rng.uniform(0, world_width), rng.uniform(0, world_height))
        return Agent(position=pos, brain=brain)
    
    # 1. Sadık kopyalar (%15) — öğrenileni koru
    for _ in range(max(1, int(pop_size * 0.15))):
        agents.append(_make_agent(shared_brain.spawn_copy(
            mutation_rate=0.01, mutation_strength=0.02, rng=rng)))
    
    # 2. Mutasyonlu kopyalar (%25) — yakın keşif
    for _ in range(max(1, int(pop_size * 0.25))):
        agents.append(_make_agent(shared_brain.spawn_copy(
            mutation_rate=0.10, mutation_strength=0.20, rng=rng)))
    
    # 3. Elite crossover (%20) — önceki neslin en iyilerini karıştır
    if prev_elites and len(prev_elites) >= 2:
        for _ in range(max(1, int(pop_size * 0.20))):
            a, b = rng.choice(len(prev_elites), size=2, replace=False)
            child = prev_elites[a].crossover(prev_elites[b], rng=rng)
            child.mutate(mutation_rate=0.05, mutation_strength=0.12, rng=rng)
            agents.append(_make_agent(child))
    else:
        # İlk nesilde elite yok, daha fazla mutant üret
        for _ in range(max(1, int(pop_size * 0.20))):
            agents.append(_make_agent(shared_brain.spawn_copy(
                mutation_rate=0.20, mutation_strength=0.35, rng=rng)))
    
    # 4. Tamamen rastgele beyinler (%20) — "yeni ruhlar"
    for _ in range(max(1, int(pop_size * 0.20))):
        fresh_brain = NeuralNetwork()  # Sıfırdan rastgele ağırlıklar
        agents.append(_make_agent(fresh_brain))
    
    # 5. Çılgın mutasyon (%20) — radikal denemeler
    while len(agents) < pop_size:
        agents.append(_make_agent(shared_brain.spawn_copy(
            mutation_rate=0.30, mutation_strength=0.50, rng=rng)))
    
    return agents[:pop_size]


def draw_ui(
    surface: pygame.Surface,
    font: pygame.font.Font,
    shared_brain: SharedBrain,
    generation: int,
    alive_count: int,
    frame_count: int,
    best_agent_fitness: float,
    last_survivor_info: str = "",
) -> None:
    """UI çizimi — SharedBrain bilgileri ön planda."""
    maturity = shared_brain.maturity
    mat_color = MATURITY_COLORS.get(maturity, TEXT_COLOR)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        # SharedBrain bilgileri (ruh)
        (f"☰ THE EGG — Faz 0: Hayatta Kalma", SOUL_COLOR),
        (f"Ruh: {maturity}  ({shared_brain.total_lives} hayat yaşandı)", mat_color),
        (f"Toplam Yem: {shared_brain.total_food}", HIGHLIGHT_COLOR),
        (f"En İyi Skor (tüm zamanlar): {shared_brain.best_fitness_ever:.0f}", HIGHLIGHT_COLOR),
        ("", TEXT_COLOR),  # boşluk
        # Nesil bilgileri
        (f"Nesil: {generation}", TEXT_COLOR),
        (f"Hayatta: {alive_count}/{POPULATION_SIZE}", TEXT_COLOR),
        (f"Bu Nesil En İyi: {best_agent_fitness:.0f}", DIM_COLOR),
        (f"Frame: {frame_count}/{GENERATION_FRAME_LIMIT}", DIM_COLOR),
    ]

    if last_survivor_info:
        lines.append(("", TEXT_COLOR))
        lines.append(("--- BATTLE ROYALE ---", (255, 100, 100)))
        lines.append((last_survivor_info, (255, 200, 100)))

    for idx, (text, col) in enumerate(lines):
        if text:
            label = font.render(text, True, col)
            surface.blit(label, (16, 14 + idx * 26))

    # Ortalama fitness trendi (sağ alt köşe)
    scores = shared_brain.generation_scores
    if len(scores) >= 2:
        trend = scores[-1] - scores[-2]
        trend_text = f"▲ {trend:+.0f}" if trend >= 0 else f"▼ {trend:.0f}"
        trend_color = (100, 255, 150) if trend >= 0 else (255, 100, 100)
        trend_label = font.render(trend_text, True, trend_color)
        surface.blit(trend_label, (WINDOW_WIDTH - 120, WINDOW_HEIGHT - 40))


def draw_fitness_graph(
    surface: pygame.Surface,
    scores: list[float],
) -> None:
    """Sağ üst köşede küçük fitness grafiği çiz."""
    if len(scores) < 2:
        return

    graph_x = WINDOW_WIDTH - 220
    graph_y = 14
    graph_w = 200
    graph_h = 80

    # Arka plan
    bg_rect = pygame.Rect(graph_x - 4, graph_y - 4, graph_w + 8, graph_h + 8)
    pygame.draw.rect(surface, (30, 32, 38), bg_rect, border_radius=6)
    pygame.draw.rect(surface, (60, 62, 68), bg_rect, 1, border_radius=6)

    # Son 50 neslin skorları
    visible = scores[-50:]
    if not visible:
        return

    max_score = max(visible) or 1.0
    min_score = min(visible)
    score_range = max(max_score - min_score, 1.0)

    points = []
    for i, score in enumerate(visible):
        x = graph_x + int(i * graph_w / max(len(visible) - 1, 1))
        y = graph_y + graph_h - int((score - min_score) / score_range * graph_h)
        points.append((x, y))

    if len(points) >= 2:
        pygame.draw.lines(surface, SOUL_COLOR, False, points, 2)


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("The Egg — Faz 0: Hayatta Kalma")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    rng = np.random.default_rng()

    # ═══════════════════════════════════════════════════════════════
    # THE EGG: Tek bir beyin — tüm hayatların ruhu
    # Önce diskten yüklemeyi dene (geçmiş hayatların hafızası)
    # ═══════════════════════════════════════════════════════════════
    shared_brain = SharedBrain.load()
    if shared_brain is not None:
        generation = len(shared_brain.generation_scores) + 1
        print(f"Ruh uyanıyor... {shared_brain.total_lives} hayat yaşanmış, {shared_brain.maturity} seviyesinde.")
    else:
        shared_brain = SharedBrain()
        generation = 1
        print("Yeni bir ruh doğuyor...")

    generation_frame = 0
    prev_elites: list[NeuralNetwork] | None = None  # Önceki neslin en iyi beyinleri

    # İlk nesil: SharedBrain'den kopyalarla başla
    agents = create_agents_from_shared_brain(
        shared_brain, POPULATION_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, rng,
        prev_elites=prev_elites,
    )
    food_manager = FoodManager(WINDOW_WIDTH, WINDOW_HEIGHT, FOOD_COUNT)

    last_survivor_info = ""  # Son hayatta kalanın bilgisi

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        generation_frame += 1

        # --- Tüm ajanları güncelle ---
        for agent in agents:
            agent.update(food_manager, agents, WINDOW_WIDTH, WINDOW_HEIGHT)

        # --- Predation: büyük olan küçüğü yer ---
        alive_agents = [a for a in agents if a.alive]
        for i, predator in enumerate(alive_agents):
            for prey in alive_agents[i + 1:]:
                if not predator.alive or not prey.alive:
                    continue
                dist = (predator.position - prey.position).length()
                touch_dist = predator.actual_radius + prey.actual_radius

                if dist > touch_dist:
                    continue

                # Büyük olan küçüğü yer
                if predator.can_eat(prey):
                    prey.alive = False
                    predator.health = min(predator.max_health, predator.health + 40.0)
                    predator.agents_eaten += 1
                    predator.grow(0.15)  # Ajan yemek daha çok büyütür
                elif prey.can_eat(predator):
                    predator.alive = False
                    prey.health = min(prey.max_health, prey.health + 40.0)
                    prey.agents_eaten += 1
                    prey.grow(0.15)

        alive_count = sum(1 for a in agents if a.alive)
        best_agent_fitness = max((a.fitness for a in agents), default=0.0)

        # Son 1 kişi kaldığında bilgisini kaydet
        if alive_count == 1:
            survivor = next(a for a in agents if a.alive)
            last_survivor_info = (
                f"Son kalan: {survivor.food_eaten} yem, "
                f"{survivor.agents_eaten} av, "
                f"boyut: {survivor.growth:.1f}x"
            )

        # --- Nesil sonu ---
        if alive_count == 0 or generation_frame >= GENERATION_FRAME_LIMIT:
            # Bu neslin en iyilerinin beyinlerini sakla (crossover için)
            sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
            elite_count = max(2, len(agents) // 5)  # Top %20
            prev_elites = [a.brain.clone() for a in sorted_agents[:elite_count]]

            # SharedBrain'i bu neslin deneyimiyle güncelle
            # Ana ödül: SON HAYATTA KALAN'ın beyni
            shared_brain.update_from_generation(agents)
            shared_brain.save()

            # Yeni nesil
            generation += 1
            generation_frame = 0
            agents = create_agents_from_shared_brain(
                shared_brain, POPULATION_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, rng,
                prev_elites=prev_elites,
            )
            food_manager.reset(FOOD_COUNT)

        # --- Render ---
        screen.fill(BACKGROUND_COLOR)
        food_manager.draw(screen)

        for agent in agents:
            agent.draw(screen)

        draw_ui(
            screen, font, shared_brain,
            generation, alive_count, generation_frame, best_agent_fitness,
            last_survivor_info
        )
        draw_fitness_graph(screen, shared_brain.generation_scores)

        pygame.display.flip()
        clock.tick(FPS)

    # Kapanırken son kez kaydet
    shared_brain.save()
    print(f"Ruh uykuya dalıyor... {shared_brain.total_lives} hayat yaşandı.")
    pygame.quit()


if __name__ == "__main__":
    run()
