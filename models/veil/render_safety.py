"""Pixel-level invariants, independent of model imports."""
import numpy as np


def restore_protected_regions(rendered, original, contexts):
    """Restore raw AND smoothed target boxes after all swap/blur composition.

    This is a target-priority policy: overlapping non-target pixels may remain
    exposed. The evaluator must report that tradeoff rather than calling it
    complete anonymization.
    """
    h, w = rendered.shape[:2]
    for ctx in contexts:
        if not ctx['is_target_final']:
            continue
        for key in ('raw_bbox', 'smoothed_bbox'):
            x1, y1, x2, y2 = map(int, ctx[key])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                rendered[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    return rendered


def changed_pixels(original, rendered, bbox):
    h, w = rendered.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(np.any(original[y1:y2, x1:x2] != rendered[y1:y2, x1:x2], axis=2).sum())
