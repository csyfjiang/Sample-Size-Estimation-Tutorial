"""
cli.py -- command-line interface for sampsizeval.

Examples
--------
  sampsizeval dev        --n 717 --phi 0.30 --P 20
  sampsizeval val-closed --n 334 --phi 0.30 --c 0.75
  sampsizeval val-sim    --n 334 --mu -0.85 --sigma 1.0
  sampsizeval compare    --n 334 --theta1 0.75 --theta2 0.82 --rho 0.5 --gamma 0.30
  sampsizeval from-csv predictions.csv         # -> phi, c-stat, mu, sigma
  sampsizeval from-csv two_models.csv --two    # -> theta1, theta2, rho, gamma
"""

import argparse
import csv
import sys

from . import development, validation_closed, validation_sim, compare_auc, data


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for key in rows[0]:
        cols[key] = [r[key] for r in rows]
    return cols


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def cmd_dev(a):
    r2 = a.r2cs if a.r2cs is not None else 0.15 * development.max_r2cs(a.phi)
    r = development.reverse_lookup_development(a.n, a.phi, a.P, r2)
    print(f"\nDEVELOPMENT  N={a.n}  phi={a.phi}  P={a.P}  R2cs={_fmt(r2)}")
    print(f"  B1 margin      = {_fmt(r['B1_margin'],4)}  (need N>={r['B1_required_n']:.0f})  {'OK' if r['B1_ok'] else 'X'}")
    print(f"  B2 MAPE        = {_fmt(r['B2_mape'],4)}  (need N>={r['B2_required_n']:.0f})  {'OK' if r['B2_ok'] else 'X'}")
    print(f"  B3 shrinkage S = {_fmt(r['B3_shrinkage'])}  (need N>={r['B3_required_n']:.0f})  {'OK' if r['B3_ok'] else 'X'}")
    print(f"  B4 optimism    = {_fmt(r['B4_optimism'],4)}  (need N>={r['B4_required_n']:.0f})  {'OK' if r['B4_ok'] else 'X'}")
    print(f"  => required N = {r['overall_required_n']:.0f}  (binding: {r['binding_criterion']})  "
          f"{'SUFFICIENT' if r['sufficient'] else 'SHORT by %d' % r['shortfall']}")


def cmd_val_closed(a):
    r = validation_closed.reverse_lookup_binary(a.n, a.phi, a.c, n_sims=a.nsims)
    print(f"\nVALIDATION (closed form)  N={a.n}  phi={a.phi}  c={a.c}")
    print(f"  O/E CI width          = {_fmt(r['oe_ci_width'])}  {'OK' if r['oe_ok'] else 'WIDE'}")
    print(f"  calibration slope CI  = {_fmt(r['calibration_slope_ci_width'])}  {'OK' if r['calibration_slope_ok'] else 'WIDE'}")
    print(f"  c-statistic CI width  = {_fmt(r['c_stat_ci_width'])}  {'OK' if r['c_stat_ok'] else 'WIDE'}")


def cmd_val_sim(a):
    r = validation_sim.reverse_lookup_simulation(a.n, a.mu, a.sigma, a.gamma, a.S, n_sims=a.nsims)
    print(f"\nVALIDATION (simulation)  N={a.n}  mu={a.mu}  sigma={a.sigma}  gamma={a.gamma}  S={a.S}")
    print(f"  mean events           = {_fmt(r['mean_events'],1)}")
    print(f"  c-statistic CI width  = {_fmt(r['c_statistic_ci_width'])}")
    print(f"  calibration slope CI  = {_fmt(r['calibration_slope_ci_width'])}")
    print(f"  O/E CI width          = {_fmt(r['oe_ci_width'])}")


def cmd_compare(a):
    fwd = compare_auc.required_sample_size(a.theta1, a.theta2, a.rho, a.gamma, a.alpha, a.power)
    print(f"\nCOMPARE AUCs  theta1={a.theta1}  theta2={a.theta2}  rho={a.rho}  gamma={a.gamma}")
    print(f"  required total N (power={a.power}) = {fwd['N']}  (cases {fwd['m']}, controls {fwd['n']})")
    if a.n:
        pw = compare_auc.achieved_power(a.n, a.theta1, a.theta2, a.rho, a.gamma, a.alpha)
        print(f"  achieved power at N={a.n} = {_fmt(pw['power'])}  [{pw['verdict']}]")


def cmd_from_csv(a):
    cols = _read_csv(a.path)
    if a.two:
        p = data.params_from_two_models(cols)
        print(f"\nFrom {a.path}  (N={p['n']}, events={p['events']})")
        print(f"  theta1 (model A AUC) = {_fmt(p['theta1'])}")
        print(f"  theta2 (model B AUC) = {_fmt(p['theta2'])}")
        print(f"  rho (score corr)     = {_fmt(p['rho'])}")
        print(f"  gamma (event prop)   = {_fmt(p['gamma'])}")
    else:
        p = data.params_from_predictions(cols)
        print(f"\nFrom {a.path}  (N={p['n']}, events={p['events']})")
        print(f"  phi (event prop)     = {_fmt(p['phi'])}")
        print(f"  c-statistic (AUC)    = {_fmt(p['c_stat'])}")
        print(f"  LP mu / sigma        = {_fmt(p['mu'])} / {_fmt(p['sigma'])}")


def build_parser():
    p = argparse.ArgumentParser(prog="sampsizeval",
                                description="Sample-size / precision calculators "
                                            "for binary-outcome prediction models.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dev", help="development-stage sample size")
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--phi", type=float, required=True)
    d.add_argument("--P", type=int, required=True)
    d.add_argument("--r2cs", type=float, default=None)
    d.set_defaults(func=cmd_dev)

    v = sub.add_parser("val-closed", help="validation precision (closed form)")
    v.add_argument("--n", type=int, required=True)
    v.add_argument("--phi", type=float, required=True)
    v.add_argument("--c", type=float, required=True)
    v.add_argument("--nsims", type=int, default=2000)
    v.set_defaults(func=cmd_val_closed)

    s = sub.add_parser("val-sim", help="validation precision (simulation)")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--mu", type=float, required=True)
    s.add_argument("--sigma", type=float, required=True)
    s.add_argument("--gamma", type=float, default=0.0)
    s.add_argument("--S", type=float, default=1.0)
    s.add_argument("--nsims", type=int, default=500)
    s.set_defaults(func=cmd_val_sim)

    c = sub.add_parser("compare", help="compare two correlated AUCs")
    c.add_argument("--theta1", type=float, required=True)
    c.add_argument("--theta2", type=float, required=True)
    c.add_argument("--rho", type=float, required=True)
    c.add_argument("--gamma", type=float, required=True)
    c.add_argument("--n", type=int, default=None)
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--power", type=float, default=0.8)
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("from-csv", help="estimate inputs from a patient-level CSV")
    f.add_argument("path")
    f.add_argument("--two", action="store_true", help="two-model comparison CSV")
    f.set_defaults(func=cmd_from_csv)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
