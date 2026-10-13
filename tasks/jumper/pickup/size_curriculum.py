"""Evidence-gated size ladder; never advance because training merely finished."""
import math

from .bootstrap import bounded_int

LEVELS = (.05, .10, .20)


def next_variation(current, results_by_size):
    """Require 8/12 placements, 10/12 pickups and zero penetration per size.

    Initial conservative experiment gates, not tuned reliability claims. Supply
    separate, previously unused evaluation seeds for every size and every gate.
    The experiment driver owns the seed reservation; this checks report integrity.
    """
    if (type(current) not in (float, int) or current not in LEVELS or
            not isinstance(results_by_size, dict) or
            set(results_by_size) != {'small', 'standard', 'large'}):
        raise ValueError('need a supported level and all three size reports')
    seeds, passed = set(), True
    for rows in results_by_size.values():
        if not isinstance(rows, list) or len(rows) != 12:
            raise ValueError('need exactly 12 distinct held-out trials per size')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('trial must be a report object')
            seed = bounded_int(row.get('seed'), 'evaluation-seed', 0, 2**31 - 1)
            depth = row.get('max_chassis_penetration_mm')
            endings = row.get('termination')
            if (seed in seeds or type(row.get('placed')) is not bool or
                    type(row.get('picked_up')) is not bool or
                    type(depth) not in (int, float) or not math.isfinite(depth) or depth < 0 or
                    not isinstance(endings, list) or any(type(s) is not str for s in endings)):
                raise ValueError('invalid or duplicate evaluation trial')
            seeds.add(seed)
            passed &= depth == 0 and not any(name in endings for name in
                ('chassis_intrusion', 'invalid_fixture', 'fell_over', 'too_low'))
        passed &= sum(row['placed'] for row in rows) >= 8
        passed &= sum(row['picked_up'] for row in rows) >= 10
    index = LEVELS.index(current)
    return LEVELS[min(index + 1, len(LEVELS) - 1)] if passed else current
