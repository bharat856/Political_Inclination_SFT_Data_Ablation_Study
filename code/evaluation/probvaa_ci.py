"""Final primary notebook cell 15: analytic 95% policy intervals, not bootstrap."""
import math


def policy_estimate(items, weighted=False):
    # Each item is (annotation sign, statement score); missing scores excluded.
    pairs = [(a, r) for a, r in items if r is not None and not math.isnan(r)]
    n = len(pairs)
    if not n:
        return {'score': None, 'coverage': 0, 'ci_halfwidth': None, 'ci_low': None, 'ci_high': None}
    x = [a * r for a, r in pairs]
    if not weighted:
        mean = sum(x) / n
        hw = 1.96 * math.sqrt(sum((v - mean)**2 for v in x) / (n - 1)) / math.sqrt(n) if n > 1 else None
    else:
        w = [abs(r) for _, r in pairs]
        total, squared = sum(w), sum(v*v for v in w)
        mean = sum(v*z for v, z in zip(w, x)) / total if total > 0 else None
        hw = (1.96 * math.sqrt(sum(v*v*(z-mean)**2 for v, z in zip(w, x)) / (total*total - squared))
              if total > 0 and total*total > squared and n > 1 else None)
    return {'score': mean, 'coverage': n, 'ci_halfwidth': hw,
            'ci_low': mean - hw if hw is not None else None,
            'ci_high': mean + hw if hw is not None else None}


def significance(score, halfwidth):
    # Cell 18, including its zero-width => no significance behavior.
    if score is None or halfwidth is None or halfwidth == 0:
        return {'p_value': None, 'significant': False}
    se = halfwidth / 1.959963984540054
    p = math.erfc(abs(score / se) / math.sqrt(2))
    return {'p_value': p, 'significant': p < 0.05}
