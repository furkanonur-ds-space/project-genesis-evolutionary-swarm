"""Sequence models whose weights come from a batch of ES genomes.

Every model has the same interface:

    spec: GenomeSpec                  layout of the genome
    forward(genomes, tokens) -> logits
        genomes: (population, spec.size)
        tokens:  (batch, time) integer input symbols, shared by all members
        logits:  (population, batch, time, num_outputs)

The model reads exactly one input symbol per time step and emits exactly
one output distribution per time step (serial read, serial write).
"""

from egg.models.colony import Colony
from egg.models.gru import GRUBaseline

__all__ = ["Colony", "GRUBaseline"]
