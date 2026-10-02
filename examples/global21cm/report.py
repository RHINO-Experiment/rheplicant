"""The tables ``analyse.py`` prints, which the README quotes."""

from __future__ import annotations


def _f(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def print_tables(reports: dict) -> None:
    """Headline, goodness of fit, samplers, and coverage, one table each."""
    print("| scenario | posterior | SER | SER/prior | var/prior | bias frac | z2 (7 dof) | depth q | "
          "position q | GoF | calibrated | eta SMC | eta Laplace | sigma(amp) | bias/sigma |")
    print("|" + "---|" * 15)
    for name, report in reports.items():
        for key in ("prior", "oracle", "beamconv", "physical"):
            e = report[key]
            fc = e.get("forecast", {})
            print(f"| {name} | {key} | {e['ser']:.3g} | {e['ser_over_prior']:.3g} | "
                  f"{e['variance_over_prior']:.3g} | {e['bias_fraction']:.2f} | {e['z2_theta']:.3g} | "
                  f"{e['depth_quantile']:.3f} | {e['position_quantile']:.3f} | "
                  f"{_f(e.get('gof', {}).get('passes'), '')} | {_f(e.get('calibrated'), '')} | "
                  f"{_f(e.get('eta_smc'), '.3g')} | {_f(e.get('eta_laplace'), '.3g')} | "
                  f"{_f(fc.get('sigma'), '.3g')} | {_f(fc.get('bias_sigma'), '+.2f')} |")
    print()
    print("| scenario | posterior | eta SMC | eta SMC core 90 % | eta Laplace | weak-signal fraction | "
          "depth q1/q5/q50/q95/q99 mK | SMC seed traces |")
    print("|" + "---|" * 8)
    for name, report in reports.items():
        for key in ("oracle", "beamconv", "physical"):
            e, t = report[key], report[key]["tail"]
            q = "/".join(f"{v:.0f}" for v in t["depth_quantiles_mk"].values())
            traces = "/".join(f"{r['trace']:.3g}" for r in e["smc"]["runs"])
            print(f"| {name} | {key} | {_f(e.get('eta_smc'), '.3g')} | {_f(e.get('eta_smc_core'), '.3g')} | "
                  f"{_f(e.get('eta_laplace'), '.3g')} | {t['weak_fraction']:.3f} | {q} | {traces} |")
    print()
    print("| scenario | posterior | chi2 compressed (dof) p | chi2 marginal (dof) p | p predictive |")
    print("|" + "---|" * 5)
    for name, report in reports.items():
        for key in ("oracle", "beamconv", "physical"):
            g = report[key]["gof"]
            print(f"| {name} | {key} | {g['chi2_compressed_best']:.1f} ({g['dof_compressed']}) "
                  f"{_f(g['p_compressed'], '.2g')} | {g['chi2_marginal']:.0f} ({g['dof_marginal']}) "
                  f"{g['p_marginal']:.2g} | {g['p_predictive']:.3f} |")
    print()
    print("| scenario | posterior | SMC log Z (seeds) | SMC s | NUTS R-hat / ESS / div. | NUTS SER/prior | "
          "spread NUTS/SMC | mean shift (SMC sd) | nu_min>0 NUTS/SMC | NUTS s |")
    print("|" + "---|" * 10)
    for name, report in reports.items():
        for key in ("oracle", "beamconv", "physical"):
            e = report[key]
            n, c = e["nuts"], e["nuts_vs_smc"]
            logz = " / ".join(f"{r['log_z']:.2f}" for r in e["smc"]["runs"])
            print(f"| {name} | {key} | {logz} | {e['smc']['seconds']:.0f} | {n['r_hat_max']:.3f} / "
                  f"{n['n_eff_min']:.0f} / {n['divergences']} | {n['ser_over_prior']:.3g} | "
                  f"{c['spread_nuts_over_smc']:.3g} | {c['mean_shift_max_sd']:.2f} | "
                  f"{c['high_nu_min_fraction']['nuts']:.2f}/{c['high_nu_min_fraction']['smc']:.3f} | "
                  f"{n['seconds']:.0f} |")
    print()
    print("| scenario | posterior | n | band cov 68 (deficit/SE) | band cov 95 (deficit/SE) | "
          "depth in 68/95 | amp within 1/2 sigma | s |")
    print("|" + "---|" * 8)
    for name, report in reports.items():
        for key in ("oracle", "beamconv", "physical"):
            v = report[key]["realisations"]
            print(f"| {name} | {key} | {v['n']} | {v['cov68']['mean']:.3f} +- {v['cov68']['se']:.3f} "
                  f"({v['cov68']['deficit_se']:+.1f}) | {v['cov95']['mean']:.3f} +- {v['cov95']['se']:.3f} "
                  f"({v['cov95']['deficit_se']:+.1f}) | {v['depth_in_central68']:.2f}/"
                  f"{v['depth_in_central95']:.2f} | {v['amplitude_within_1sigma']:.2f}/"
                  f"{v['amplitude_within_2sigma']:.2f} | {v['seconds']:.0f} |")
