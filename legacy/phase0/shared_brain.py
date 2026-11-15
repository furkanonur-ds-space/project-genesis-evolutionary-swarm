"""
SharedBrain — "The Egg" Hafızası

Mantık:
  Tüm ajanlar bu beyinden doğar (kopyasını alır).
  Her ajan biraz farklı mutasyonla dünyaya gelir.
  Nesil sonunda en iyi performans gösterenler SharedBrain'i günceller.
  
  Tıpkı "The Egg"deki gibi: her hayat, tek bir ruhu besler.
  
  Güncelleme yöntemi: Weighted Merge
  - Mevcut SharedBrain ağırlıkları + en iyi ajanın ağırlıkları karıştırılır
  - Oran: %70 en iyi ajan, %30 mevcut beyin (stabiliteyı korumak için)
  - Bu sayede beyin ani sıçramalar yapmaz, yavaşça "olgunlaşır"
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from brain import NeuralNetwork, NetworkShape

# Kayıt dosyası — beyin "uyuduğunda" hafızasını buraya yazar
SAVE_PATH = "shared_brain.pkl"


class SharedBrain:
    """Tek bir beyin, tüm ajanlar tarafından paylaşılır."""

    def __init__(self, shape: NetworkShape | None = None) -> None:
        self.shape = shape or NetworkShape()
        self.brain = NeuralNetwork(shape=self.shape)

        # İstatistikler
        self.total_lives = 0        # Kaç ajan bu beyinle yaşadı
        self.total_food = 0         # Toplam yenen yem
        self.best_fitness_ever = 0.0
        self.generation_scores: list[float] = []  # Her neslin en iyi skoru

    def spawn_copy(self, mutation_rate: float = 0.03, mutation_strength: float = 0.1,
                   rng: np.random.Generator | None = None) -> NeuralNetwork:
        """SharedBrain'den hafif mutasyonlu bir kopya üret.
        
        Her ajan aynı "ruhtan" gelir ama biraz farklıdır.
        Tıpkı aynı insanın farklı hayatlarda farklı deneyimler yaşaması gibi.
        """
        copy = self.brain.clone()
        copy.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength, rng=rng)
        return copy

    def update_from_generation(self, agents: list, merge_ratio: float = 0.7) -> None:
        """Nesil sonunda en iyi ajanların deneyimiyle güncelle.
        
        merge_ratio: Ne kadar "yeni bilgiyi" kabul et
          0.7 = %70 en iyi ajandan, %30 mevcut beyinden
          Yüksek = hızlı öğrenme ama kararsız
          Düşük = yavaş öğrenme ama kararlı
        """
        if not agents:
            return

        # En iyi ajanı bul
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        best = sorted_agents[0]

        # İstatistikleri güncelle
        self.total_lives += len(agents)
        self.total_food += sum(a.food_eaten for a in agents)
        self.best_fitness_ever = max(self.best_fitness_ever, best.fitness)
        self.generation_scores.append(best.fitness)

        # Ağırlık birleştirme (weighted merge)
        # Mevcut beyin DNA'sı + en iyi ajanın DNA'sı
        current_dna = self.brain.get_dna()
        best_dna = best.brain.get_dna()

        # İlk 3 nesilde tamamen en iyi ajandan öğren (soğuk başlangıç)
        # Sonra yavaşça merge_ratio'ya geç
        gen_count = len(self.generation_scores)
        if gen_count <= 3:
            effective_ratio = 1.0  # İlk nesillerde tamamen en iyiden kopyala
        else:
            effective_ratio = merge_ratio

        merged_dna = effective_ratio * best_dna + (1.0 - effective_ratio) * current_dna

        # Top 3'ün ortalamasını da karıştır (çeşitlilik)
        if len(sorted_agents) >= 3:
            top3_dna = np.mean([a.brain.get_dna() for a in sorted_agents[:3]], axis=0)
            # %10 top3 ortalaması ekle
            merged_dna = 0.9 * merged_dna + 0.1 * top3_dna

        self.brain = NeuralNetwork.from_dna(merged_dna, shape=self.shape)

    @property
    def maturity(self) -> str:
        """Beynin olgunluk seviyesi — kaç hayat yaşadı."""
        if self.total_lives < 100:
            return "Yenidoğan"
        elif self.total_lives < 500:
            return "Bebek"
        elif self.total_lives < 2000:
            return "Çocuk"
        elif self.total_lives < 5000:
            return "Genç"
        else:
            return "Olgun"

    @property
    def avg_recent_fitness(self) -> float:
        """Son 5 neslin ortalama fitness'ı."""
        if not self.generation_scores:
            return 0.0
        recent = self.generation_scores[-5:]
        return sum(recent) / len(recent)

    # ------------------------------------------------------------------
    # Kayıt / Yükleme — Ruh uyuduğunda hafızasını kaybetmez
    # ------------------------------------------------------------------

    def save(self, path: str = SAVE_PATH) -> None:
        """Beyni diske kaydet. Program kapansa bile hafıza korunur."""
        data = {
            "dna": self.brain.get_dna(),
            "shape": (self.brain.shape.input_size, self.brain.shape.hidden_size, self.brain.shape.output_size),
            "total_lives": self.total_lives,
            "total_food": self.total_food,
            "best_fitness_ever": self.best_fitness_ever,
            "generation_scores": self.generation_scores,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str = SAVE_PATH) -> "SharedBrain | None":
        """Diskten beyni yükle. Dosya yoksa None döner."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            shape = NetworkShape(
                input_size=data["shape"][0],
                hidden_size=data["shape"][1],
                output_size=data["shape"][2],
            )
            sb = cls(shape=shape)
            sb.brain = NeuralNetwork.from_dna(data["dna"], shape=shape)
            sb.total_lives = data["total_lives"]
            sb.total_food = data["total_food"]
            sb.best_fitness_ever = data["best_fitness_ever"]
            sb.generation_scores = data["generation_scores"]
            return sb
        except Exception as e:
            print(f"Beyin yüklenemedi: {e}")
            return None

