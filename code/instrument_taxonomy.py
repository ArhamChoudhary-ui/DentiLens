"""Shared names for the benchmark and instrument suggestions."""
from pathlib import Path

import yaml

TAXONOMY = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / 'configs/instrument_taxonomy.yaml').read_text()
)
LABELS = TAXONOMY['classes']


def canonical(raw, source):
    return TAXONOMY['source_mappings'].get(source, {}).get(
        raw, TAXONOMY['common_source_aliases'].get(raw, 'unknown')
    )
