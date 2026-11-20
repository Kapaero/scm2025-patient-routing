"""Реконструкция v3: семейство механизмов провала RL на незнакомых случаях (гипотезы о потерянном эксперименте).

Режимы (mode):
  detour          - незнакомое состояние: действие по умолчанию = длинный обход (маршрут 0)
  random          - незнакомое состояние: случайный маршрут
  hold_unseen     - незнакомое состояние: агент ждёт lat минут (колеблется) и идёт по умолчанию (прямой коридор 1)
  latency_all     - каждое решение RL сопровождается задержкой lat минут (колебание на каждом маркере)
  latency_detour  - задержка lat на каждом решении RL плюс обход на незнакомом состоянии
Все режимы: табличный RL по выборке GOAP, k-средних перед RL, цели и критерии как в v2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIT = dict(
    n_corr=4,
    length=(0.0, 4.0, 6.0, 8.0),
    headway=(0.3, 1.0, 1.5, 2.0),
    detour_len=45.0,
    dist=(0.6, 1.0, 1.7),
    p_target=(0.4, 0.4, 0.2),
    boxes=8,
    treat=(2.0, 5.0),
    crit_share=0.3,
    lam_train=0.55,
    lam_test=1.5,
    horizon=200.0,
    dt=0.25,
    qbin=2,
    kmeans_k=5,
    quantile=0.95,
    noise=0.3,
    len_scale=1.0,
    hw_scale=1.0,
    mode="detour",
    lat=10.0,
    max_tries=5,
)


@dataclass
class Agent:
    t_arr: float
    crit: bool
    target: int
    treat_len: float
    jit: float
    prio: bool = False
    action: int = -1
    t_ready: float = 0.0
    tries: int = 0
    redo: bool = False
    t_exit: float = -1.0
    t_treat: float = -1.0
    state: tuple = ()


class Sim:
    def __init__(self, lam, seed, policy, p=FIT):
        self.p = p
        rng = np.random.default_rng(seed)
        self.rng = rng
        n = int(rng.poisson(lam * p["horizon"]))
        times = np.sort(rng.uniform(0, p["horizon"], n))
        self.agents = [Agent(float(t), bool(rng.random() < p["crit_share"]), int(rng.choice(3, p=p["p_target"])),
                             float(rng.uniform(*p["treat"])), float(rng.lognormal(0.0, p["noise"]))) for t in times]
        self.policy = policy

    def length(self, c, target):
        if c == 0:
            return self.p["detour_len"]
        return self.p["length"][c] * self.p["len_scale"] * self.p["dist"][target]

    def run(self):
        p = self.p
        nc = p["n_corr"]
        hw = [p["headway"][0]] + [h * p["hw_scale"] for h in p["headway"][1:]]
        queues = [[] for _ in range(nc)]
        delayed = []
        next_free = [0.0] * nc
        walking, box_q = [], []
        box_free = [0.0] * p["boxes"]
        t, i, n, done = 0.0, 0, len(self.agents), 0
        while done < n and t < p["horizon"] * 6:
            while i < n and self.agents[i].t_arr <= t:
                a = self.agents[i]
                i += 1
                qlen = [len(q) for q in queues]
                n_sys = sum(qlen) + len(walking) + len(box_q) + len(delayed)
                a.state = (a.crit, a.target, tuple(min(q // p["qbin"], 50) for q in qlen))
                feat = np.array([float(a.crit), float(a.target)] + [float(q) for q in qlen] + [float(n_sys)])
                c, prio, delay = self.policy.choose(a, queues, qlen, feat, self, hw)
                a.action, a.prio = c, prio
                if delay > 0:
                    a.t_ready = t + delay
                    delayed.append(a)
                else:
                    queues[c].append(a)
            if delayed:
                for a in [x for x in delayed if x.t_ready <= t]:
                    delayed.remove(a)
                    if a.redo:                                   # повторное решение после ожидания: состояние пересчитывается
                        a.redo = False
                        qlen = [len(q) for q in queues]
                        n_sys = sum(qlen) + len(walking) + len(box_q) + len(delayed)
                        a.state = (a.crit, a.target, tuple(min(q // p["qbin"], 50) for q in qlen))
                        feat = np.array([float(a.crit), float(a.target)] + [float(q) for q in qlen] + [float(n_sys)])
                        c, prio, delay = self.policy.choose(a, queues, qlen, feat, self, hw)
                        a.action, a.prio = c, prio
                        if delay > 0:
                            a.t_ready = t + delay
                            delayed.append(a)
                            continue
                    queues[a.action].append(a)
            for c in range(nc):
                if queues[c] and t >= next_free[c]:
                    q = queues[c]
                    k = 0
                    cand = [j for j, x in enumerate(q) if x.prio]
                    if cand:
                        k = max(cand, key=lambda j: (q[j].crit, -q[j].t_arr))
                    a = q.pop(k)
                    a.t_exit = t + self.length(c, a.target) * a.jit
                    walking.append(a)
                    next_free[c] = t + hw[c]
            for a in [x for x in walking if x.t_exit <= t]:
                walking.remove(a)
                box_q.append(a)
            for b in range(p["boxes"]):
                if box_free[b] <= t and box_q:
                    cand = [j for j, x in enumerate(box_q) if x.prio and x.crit]
                    a = box_q.pop(cand[0] if cand else 0)
                    a.t_treat = t
                    box_free[b] = t + a.treat_len
                    done += 1
            t += p["dt"]
        return self.agents


class Goap:
    def __init__(self, eps=0.0, rng=None):
        self.eps, self.rng = eps, rng

    def choose(self, a, queues, qlen, feat, sim, hw):
        p = sim.p
        if self.eps and self.rng.random() < self.eps:
            return int(self.rng.integers(p["n_corr"])), True, 0.0
        best, bc = 1e18, 0
        for c in range(p["n_corr"]):
            ahead = sum(1 for x in queues[c] if x.crit) if a.crit else len(queues[c])
            est = ahead * hw[c] + sim.length(c, a.target)
            if est < best:
                best, bc = est, c
        return bc, True, 0.0


class TabularRL:
    def __init__(self, p=FIT):
        self.p, self.sum, self.cnt = p, {}, {}
        self.rng = np.random.default_rng(77)

    def fit(self, agents):
        for a in agents:
            if a.t_treat < 0:
                continue
            k = (a.state, a.action)
            self.sum[k] = self.sum.get(k, 0.0) + (a.t_treat - a.t_arr)
            self.cnt[k] = self.cnt.get(k, 0) + 1

    def choose(self, a, queues, qlen, feat, sim, hw):
        p = self.p
        mode = p["mode"]
        known = [(self.sum[(a.state, c)] / self.cnt[(a.state, c)], c) for c in range(p["n_corr"]) if (a.state, c) in self.cnt]
        lat = p["lat"] if mode in ("latency_all", "latency_detour") else 0.0
        if known:
            return min(known)[1], False, lat
        if mode in ("detour", "latency_detour"):
            return 0, False, lat
        if mode == "random":
            return int(self.rng.integers(p["n_corr"])), False, 0.0
        if mode == "hold_unseen":
            return 1, False, p["lat"]
        if mode == "hold_retry":
            if a.tries < p["max_tries"]:
                a.tries += 1
                a.redo = True
                return 1, False, p["lat"]
            return 1, False, 0.0
        return 1, False, lat


class Hybrid:
    def __init__(self, rl, goap, C, mean, sd, tau):
        self.rl, self.goap, self.C, self.mean, self.sd, self.tau = rl, goap, C, mean, sd, tau
        self.flagged = self.total = 0

    def choose(self, a, queues, qlen, feat, sim, hw):
        d = float(np.min(np.linalg.norm(self.C - (feat - self.mean) / self.sd, axis=1)))
        self.total += 1
        if d > self.tau:
            self.flagged += 1
            return self.goap.choose(a, queues, qlen, feat, sim, hw)
        return self.rl.choose(a, queues, qlen, feat, sim, hw)


def kmeans(X, k, rng):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(60):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def build(p=FIT, seed=1000):
    rng = np.random.default_rng(seed)
    sample, feats = [], []
    for e in range(25):
        g = Goap(0.35, rng)

        class Rec:
            def choose(self, a, queues, qlen, feat, sim, hw, g=g):
                feats.append(feat)
                return g.choose(a, queues, qlen, feat, sim, hw)

        sample += Sim(p["lam_train"], seed + e, Rec(), p).run()
    rl = TabularRL(p)
    rl.fit(sample)
    X = np.array(feats)
    mean, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / sd
    C = kmeans(Z, p["kmeans_k"], rng)
    d = np.min(np.linalg.norm(Z[:, None, :] - C[None], axis=2), axis=1)
    return rl, C, mean, sd, float(np.quantile(d, p["quantile"]))


def stats(agents):
    cr = np.array([a.t_treat - a.t_arr for a in agents if a.crit and a.t_treat >= 0])
    rg = np.array([a.t_treat - a.t_arr for a in agents if (not a.crit) and a.t_treat >= 0])
    return cr, rg


def run_case(seed, built, p=FIT):
    rl, C, mean, sd, tau = built
    out = {}
    for name in ("GOAP", "RL", "RL+кластеры+GOAP"):
        pol = Goap() if name == "GOAP" else (rl if name == "RL" else Hybrid(rl, Goap(), C, mean, sd, tau))
        ag = Sim(p["lam_test"], seed, pol, p).run()
        out[name] = (stats(ag), getattr(pol, "flagged", 0) / max(1, getattr(pol, "total", 1)))
    return out
