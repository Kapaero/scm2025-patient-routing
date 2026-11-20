"""Повтор эксперимента старой статьи (сети Петри, GOAP, генетический алгоритм) в модели коридоров v3 без возврата.
Запуск: python fit_v10.py <seed> <n>"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_v3 as F3  # noqa: E402
import reconstruct_v3 as R  # noqa: E402

BASE = json.loads(Path(__file__).with_name("_v3_detour.json").read_text(encoding="utf-8"))[1]
T = {"Петри": ((25.0, 5.5), (25.0, 5.5)), "GOAP": ((14.0, 6.2), (21.0, 6.5)), "Генетика": ((44.0, 10.0), (31.0, 10.0))}
MAPS = ([1, 2, 3], [2, 2, 2], [1, 1, 2], [2, 3, 3])      # номер коридора по цели (сети Петри: жёсткая последовательность)


class GoapD(R.Goap):
    """GOAP с временем планирования на каждом решении."""

    def __init__(self, delay=0.0):
        super().__init__()
        self.delay = delay

    def choose(self, a, queues, qlen, feat, sim, hw):
        c, pr, _ = super().choose(a, queues, qlen, feat, sim, hw)
        return c, pr, self.delay


class Petri:
    def __init__(self, m, delay=0.0):
        self.m = MAPS[int(m) % len(MAPS)]
        self.delay = delay

    def choose(self, a, queues, qlen, feat, sim, hw):
        return self.m[a.target], False, self.delay


class GA:
    """Выбор коридора по приспособленности со штрафом, который не привязан к действию уступки; приоритета в очереди нет."""

    def __init__(self, lam, eps, seed, delay=0.0):
        self.lam, self.eps, self.rng, self.delay = lam, eps, np.random.default_rng(seed), delay

    def choose(self, a, queues, qlen, feat, sim, hw):
        p = sim.p
        if self.rng.random() < self.eps:
            return int(self.rng.integers(p["n_corr"])), False, self.delay
        best, bc = 1e18, 0
        for c in range(p["n_corr"]):
            est = len(queues[c]) * hw[c] + sim.length(c, a.target)
            if a.crit:
                est += self.lam * (len(queues[c]) + 1) * hw[c] * 6.0
            if est < best:
                best, bc = est, c
        return bc, False, self.delay


def make(params):
    q = dict(BASE)
    q.update(params)
    return F3.make(q, "detour")


def run(params, seeds):
    p = make(params)
    acc = {n: ([], []) for n in T}
    for s in seeds:
        for n, pol in (("Петри", Petri(params["petri_map"], params.get("pet_delay", 0.0))), ("GOAP", GoapD(params.get("goap_delay", 0.0))), ("Генетика", GA(params["ga_lam"], params["ga_eps"], s, params.get("ga_delay", 0.0)))):
            ag = R.Sim(p["lam_test"], s, pol, p).run()
            cr, rg = R.stats(ag)
            acc[n][0].append(cr)
            acc[n][1].append(rg)
    return {n: (np.concatenate(c), np.concatenate(g)) for n, (c, g) in acc.items()}


def evaluate(params):
    res = run(params, range(1, 11))
    e, rows = 0.0, {}
    for n in T:
        for k in (0, 1):
            x = res[n][k]
            m, sd = (x.mean(), x.std()) if len(x) else (999.0, 99.0)
            tm, ts = T[n][k]
            e += (2.0 if n == 'GOAP' else 1.0) * (((m - tm) / tm) ** 2 + 0.5 * ((sd - ts) / ts) ** 2)
            rows[f"{n}:{'крит' if k == 0 else 'ост'}"] = (round(float(m), 1), round(float(sd), 1))
    return e, params, rows


def sample(rng):
    return dict(lam_test=float(rng.uniform(0.6, 1.8)), crit_share=float(rng.uniform(0.25, 0.55)), len_scale=float(rng.choice([1.5, 2.0, 3.0, 3.4])),
                hw_scale=float(rng.choice([0.6, 0.7, 0.8, 1.0])), noise=float(rng.uniform(0.0, 0.6)), ga_lam=float(rng.uniform(0.0, 3.0)),
                ga_eps=float(rng.uniform(0.0, 0.3)), petri_map=int(rng.integers(len(MAPS))), pet_delay=float(rng.uniform(0.0, 18.0)), goap_delay=float(rng.uniform(0.0, 10.0)), ga_delay=float(rng.uniform(0.0, 18.0)))


def refine(top, rng):
    b = dict(top[int(rng.integers(len(top)))][1])
    for k, lo, hi in (("lam_test", 0.9, 1.1), ("crit_share", 0.9, 1.1), ("len_scale", 0.9, 1.1), ("noise", 0.7, 1.3), ("ga_lam", 0.8, 1.25), ("ga_eps", 0.8, 1.25), ("hw_scale", 0.9, 1.1), ("pet_delay", 0.85, 1.15), ("goap_delay", 0.85, 1.15), ("ga_delay", 0.85, 1.15)):
        b[k] = float(b[k] * rng.uniform(lo, hi))
    return b


if __name__ == "__main__":
    seed, n = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(seed)
    with ProcessPoolExecutor() as ex:
        out = sorted(ex.map(evaluate, [sample(rng) for _ in range(n)]), key=lambda x: x[0])
        for _ in range(4):
            more = list(ex.map(evaluate, [refine(out[:4], rng) for _ in range(n // 2)]))
            out = sorted(out + more, key=lambda x: x[0])
    e, prm, rows = out[0]
    print(f"[v10] ошибка {e:.3f}")
    for k, v in rows.items():
        print("   ", k, v)
    Path(__file__).with_name("_v10.json").write_text(json.dumps([e, prm, rows], ensure_ascii=False), encoding="utf-8")
