"""Подбор для каждого механизма провала RL. Запуск: python fit_v3.py <mode> <seed> <n>"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_v3 as R  # noqa: E402

T = {
    "GOAP": ((14.0, 6.2, 6.0, 26.0), (21.0, 6.5, 11.0, 32.0)),
    "RL": ((54.0, 6.0, 43.0, 64.0), (50.0, 6.0, 42.0, 60.0)),
    "RL+кластеры+GOAP": ((19.0, 6.5, 10.0, 31.0), (23.0, 6.5, 13.0, 33.0)),
}
W = (1.0, 0.5, 0.25, 0.25)


def make(params, mode):
    p = dict(R.FIT)
    p.update(params)
    p["mode"] = mode
    p["dist"] = (params.get("d0", 0.6), 1.0, params.get("d2", 1.7))
    w = np.array([params.get("w0", 0.4), params.get("w1", 0.4), params.get("w2", 0.2)])
    p["p_target"] = tuple(w / w.sum())
    return p


def evaluate(arg):
    params, mode = arg
    p = make(params, mode)
    built = R.build(p)
    errs, rows = 0.0, {}
    acc = {n: ([], []) for n in T}
    for seed in range(1, 5):
        res = R.run_case(seed, built, p)
        for n in T:
            (cr, rg), _ = res[n]
            acc[n][0].append(cr)
            acc[n][1].append(rg)
    # контроль: при знакомой нагрузке RL не должен быть хуже GOAP (условие сюжета статьи)
    pc = dict(p)
    pc["lam_test"] = p["lam_train"]
    ctrl = []
    for seed in (1, 2):
        rc = R.run_case(seed, built, pc)
        ctrl.append((np.mean(np.concatenate([rc["RL"][0][0], rc["RL"][0][1]])), np.mean(np.concatenate([rc["GOAP"][0][0], rc["GOAP"][0][1]]))))
    cr_, cg_ = np.mean([c[0] for c in ctrl]), np.mean([c[1] for c in ctrl])
    errs += 3.0 * max(0.0, (cr_ - 1.25 * cg_) / cg_) ** 2
    rows["контроль RL/GOAP"] = (round(float(cr_), 1), round(float(cg_), 1))
    for n in T:
        for k in (0, 1):
            x = np.concatenate(acc[n][k])
            got = (x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95))
            for g, t_, w in zip(got, T[n][k], W):
                errs += w * ((g - t_) / t_) ** 2
            rows[f"{n}:{'крит' if k == 0 else 'ост'}"] = tuple(round(float(g), 1) for g in got)
    return errs, params, rows


def sample(rng):
    return dict(lam_test=float(rng.uniform(1.0, 2.4)), lam_train=float(rng.choice([0.4, 0.55, 0.8])), crit_share=float(rng.uniform(0.2, 0.55)),
                boxes=int(rng.choice([6, 8, 10])), hw_scale=float(rng.choice([0.6, 0.8, 1.0])), len_scale=float(rng.choice([1.0, 1.5, 2.0, 3.0])),
                quantile=float(rng.uniform(0.85, 0.999)), qbin=int(rng.choice([1, 2, 3])), noise=float(rng.uniform(0.0, 0.8)),
                detour_len=float(rng.uniform(25.0, 65.0)), lat=float(rng.uniform(4.0, 40.0)), max_tries=int(rng.integers(2, 9)),
                d0=float(rng.uniform(0.3, 0.9)), d2=float(rng.uniform(1.3, 3.0)), w0=float(rng.uniform(0.2, 0.6)), w1=float(rng.uniform(0.2, 0.6)), w2=float(rng.uniform(0.05, 0.4)))


def refine(top, rng):
    b = dict(top[int(rng.integers(len(top)))][1])
    for k, lo, hi in (("lam_test", 0.9, 1.1), ("crit_share", 0.85, 1.15), ("len_scale", 0.9, 1.1), ("noise", 0.7, 1.3), ("d0", 0.9, 1.1), ("d2", 0.9, 1.1),
                      ("w0", 0.8, 1.2), ("w2", 0.8, 1.2), ("hw_scale", 0.9, 1.1), ("detour_len", 0.9, 1.1), ("lat", 0.8, 1.2)):
        b[k] = float(b[k] * rng.uniform(lo, hi))
    b["quantile"] = float(rng.uniform(0.85, 0.9995))
    return b


if __name__ == "__main__":
    mode, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(evaluate, [(sample(rng), mode) for _ in range(n)]))
        out.sort(key=lambda x: x[0])
        for _ in range(5):                          # два уточняющих этапа
            top = out[:4]
            more = list(ex.map(evaluate, [(refine(top, rng), mode) for _ in range(n // 2)]))
            out = sorted(out + more, key=lambda x: x[0])
    e, prm, rows = out[0]
    print(f"[{mode}] ошибка {e:.3f}")
    for k, v in rows.items():
        print("   ", k, v)
    Path(__file__).with_name(f"_v3_{mode}.json").write_text(json.dumps([e, prm, rows], ensure_ascii=False), encoding="utf-8")


