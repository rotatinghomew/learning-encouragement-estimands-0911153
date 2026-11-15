"""Household-clustered encouragement-design analysis.

Two estimands are kept visibly separate:

* ``invitation_effect`` -- the effect of *being offered* the course.  Household
  randomisation identifies it, and it is the only causal quantity the design
  identifies without extra assumptions.
* ``participation_effect`` -- the effect of *actually attending*.  It requires a
  complier structure, so it is reported as partially identified bounds plus
  declared sensitivity scenarios.  No bare ratio of two mean differences is
  presented as a benefit.

Households are the replication unit: members share the invitation and are never
counted as independent observations.  Missing follow-up stays missing; it
widens the bounds and is never replaced by zero.
"""

import math
from collections import defaultdict

import numpy as np

# Declared defaults.  Every one is echoed in the saved result so a later reader
# can replay the analysis; none is a fact identified by the data.
DEFAULT_DEFIER_RATE = 0.0
DEFAULT_DEFIER_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
DEFAULT_SOCIAL_SPILLOVER = 0.0
DEFAULT_SPILLOVER_GRID = (0.0, 0.25, 0.50, 1.0)
DEFAULT_DIRECTION = "same"
DEFAULT_SEED = 20261115
DEFAULT_REPLICATES = 2000
DEFAULT_CONFIDENCE = 0.95
MIN_VALID_RESAMPLES = 0.90

_TOLERANCE = 1e-9


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    return float(value)


def _probability(value, name):
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}: expected a probability in [0, 1]")
    return value


def parse_options(options):
    """Validate caller assumptions and make every default explicit."""
    options = {} if options is None else dict(options)
    allowed = {
        "defier_rate", "defier_grid", "social_spillover", "spillover_grid",
        "spillover_direction", "outcome_bounds", "seed", "bootstrap_replicates",
        "confidence",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("unknown option(s): " + ", ".join(sorted(unknown)))

    def grid(name, default):
        value = options.get(name, default)
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError(f"{name}: expected a list of probabilities")
        if not value:
            raise ValueError(f"{name}: expected a non-empty list")
        parsed = [_probability(item, f"{name}[]") for item in value]
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"{name}: duplicate values")
        return sorted(parsed)

    direction = options.get("spillover_direction", DEFAULT_DIRECTION)
    if direction not in ("same", "opposite"):
        raise ValueError("spillover_direction: expected 'same' or 'opposite'")

    seed = options.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed: expected an integer")
    replicates = options.get("bootstrap_replicates", DEFAULT_REPLICATES)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 1 <= replicates <= 100000:
        raise ValueError("bootstrap_replicates: expected an integer in [1, 100000]")
    confidence = _probability(options.get("confidence", DEFAULT_CONFIDENCE), "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence: expected a probability strictly inside (0, 1)")

    declared_bounds = options.get("outcome_bounds")
    if declared_bounds is not None:
        if isinstance(declared_bounds, (str, bytes)) or not isinstance(declared_bounds, (list, tuple)) \
                or len(declared_bounds) != 2:
            raise ValueError("outcome_bounds: expected [minimum, maximum]")
        low = _finite(declared_bounds[0], "outcome_bounds[0]")
        high = _finite(declared_bounds[1], "outcome_bounds[1]")
        if not low < high:
            raise ValueError("outcome_bounds: minimum must be below maximum")
        declared_bounds = [low, high]

    return {
        "defier_rate": _probability(options.get("defier_rate", DEFAULT_DEFIER_RATE), "defier_rate"),
        "defier_grid": grid("defier_grid", DEFAULT_DEFIER_GRID),
        "social_spillover": _probability(options.get("social_spillover", DEFAULT_SOCIAL_SPILLOVER),
                                         "social_spillover"),
        "spillover_grid": grid("spillover_grid", DEFAULT_SPILLOVER_GRID),
        "spillover_direction": direction,
        "outcome_bounds": declared_bounds,
        "seed": seed,
        "bootstrap_replicates": replicates,
        "confidence": confidence,
    }


def _check_design(packet):
    randomization = packet.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("participation_effect requires a declared randomisation design")
    if randomization.get("unit") != "household":
        raise ValueError("participation_effect requires household-level randomisation")
    mechanism = randomization.get("mechanism")
    if mechanism is not None and mechanism != "independent_bernoulli_within_stratum":
        raise ValueError("participation_effect requires independent Bernoulli assignment within strata")
    for household in packet["households"]:
        if "stratum" not in household:
            raise ValueError("participation_effect requires a stratum for every household")


def _household_records(packet):
    members = defaultdict(list)
    for person in packet["people"]:
        members[person["household"]].append(person)
    records = []
    # Canonical order makes the result independent of the order records happen
    # to appear in the source file, including the bootstrap stream.
    for household in sorted(packet["households"], key=lambda row: row["household"]):
        people = sorted(members[household["household"]], key=lambda row: row["person"])
        observed = [person["outcome"] for person in people if person["outcome"] is not None]
        attended = sum(int(person["attended"]) for person in people)
        baselines = [person.get("baseline") for person in people
                     if person.get("baseline") is not None]
        records.append({
            "id": household["household"],
            "stratum": household["stratum"],
            "invited": int(household["invited"]),
            "probability": float(household["invite_probability"]),
            "members": len(people),
            "attended": attended,
            "observed": len(observed),
            "missing": len(people) - len(observed),
            "outcome_sum": float(sum(observed)),
            "baseline": (sum(baselines) / len(baselines)) if baselines else None,
        })
    return records


def _outcome_scale(packet, declared):
    observed = [person["outcome"] for person in packet["people"] if person["outcome"] is not None]
    if declared is not None:
        low, high = declared
        if observed and (min(observed) < low - _TOLERANCE or max(observed) > high + _TOLERANCE):
            raise ValueError("outcome_bounds: declared range excludes observed outcomes")
        return low, high, "declared"
    if observed:
        return float(min(observed)), float(max(observed)), "observed_range"
    return None, None, "unavailable"


def _prepare(records, scale):
    """Attach household-level quantities used by every estimator."""
    low, high = scale
    usable = []
    for row in records:
        members = row["members"]
        if not members:
            continue
        row["baseline_sum"] = (row["baseline"] * members) if row["baseline"] is not None else None
        if low is None:
            row["outcome_sum_low"] = row["outcome_sum_high"] = None
        else:
            # Missing members stay unknown; worst cases place them at the ends
            # of the outcome scale.  They are never treated as zero.
            row["outcome_sum_low"] = row["outcome_sum"] + low * row["missing"]
            row["outcome_sum_high"] = row["outcome_sum"] + high * row["missing"]
        usable.append(row)
    return usable


def _weight(row):
    arm_probability = row["probability"] if row["invited"] else 1.0 - row["probability"]
    return 1.0 / arm_probability


def _ordered(records):
    """Stable household order so resampling does not depend on file order."""
    return sorted(records, key=lambda row: (row["stratum"], row["invited"], row["id"]))


# Each estimand is a ratio of weighted household totals.  Keeping numerator and
# denominator separate makes the unit explicit: attendance and bounds are per
# resident, while outcomes average over residents with observed follow-up.
RATIOS = {
    "attendance": ("attended", "members"),
    "outcome": ("outcome_sum", "observed"),
    "outcome_low": ("outcome_sum_low", "members"),
    "outcome_high": ("outcome_sum_high", "members"),
    "baseline": ("baseline_sum", "members"),
}


def _ratio(rows, numerator_field, denominator_field):
    numerator = denominator = 0.0
    for row in rows:
        weight = _weight(row)
        value = row[numerator_field]
        count = row[denominator_field]
        if value is None or not count:
            continue
        numerator += weight * value
        denominator += weight * count
    return numerator / denominator if denominator else None


def _stratum_arm_means(records):
    """Hajek inverse-probability ratios within each (stratum, arm).

    The weights make the arms represent the same randomised population even
    when realised assignment counts differ.  Rows remain households throughout.
    """
    groups = defaultdict(list)
    for row in _ordered(records):
        groups[(row["stratum"], row["invited"])].append(row)
    means = {}
    for key, rows in groups.items():
        for field, (numerator, denominator) in RATIOS.items():
            means[(key[0], key[1], field)] = _ratio(rows, numerator, denominator)
    return groups, means


def _stratum_sizes(records):
    """Household counts drive diagnostics; member counts drive combination."""
    households = defaultdict(lambda: {0: 0, 1: 0})
    members = defaultdict(lambda: {0: 0, 1: 0})
    for row in records:
        households[row["stratum"]][row["invited"]] += 1
        members[row["stratum"]][row["invited"]] += row["members"]
    return households, members


def _combine(means, sizes, arm, field):
    """Combine stratum means using stratum member counts.

    Allocatable stratum sizes are fixed by the design, so this averages over
    the same population the randomisation covers.
    """
    numerator = denominator = 0.0
    for stratum, arms in sizes.items():
        if not arms[0] or not arms[1]:
            continue
        value = means.get((stratum, arm, field))
        if value is None:
            continue
        weight = arms[0] + arms[1]
        numerator += weight * value
        denominator += weight
    return numerator / denominator if denominator else None


def _bootstrap_draws(groups, replicates, seed):
    """Cluster bootstrap: resample households within each stratum and arm.

    Whole households are resampled, so co-residents move together and the
    interval reflects household-level assignment.  Each replicate recomputes
    the weighted ratio, never an average of person-level draws.
    """
    rng = np.random.default_rng(seed)
    draws = {}
    for key, rows in groups.items():
        indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
        weights = np.array([_weight(row) for row in rows])
        sampled_weights = weights[indices]
        for field, (numerator_field, denominator_field) in RATIOS.items():
            numerator_values = np.array(
                [np.nan if row[numerator_field] is None else row[numerator_field]
                 for row in rows], dtype=float)
            denominator_values = np.array([row[denominator_field] for row in rows], dtype=float)
            sampled_numerator = numerator_values[indices]
            sampled_denominator = denominator_values[indices]
            mask = np.isfinite(sampled_numerator) & (sampled_denominator > 0)
            numerator = np.where(mask, sampled_weights * sampled_numerator, 0.0).sum(axis=1)
            denominator = np.where(mask, sampled_weights * sampled_denominator, 0.0).sum(axis=1)
            result = np.full(replicates, np.nan)
            np.divide(numerator, denominator, out=result, where=denominator > 0)
            draws[(key[0], key[1], field)] = result
    return draws


def _combine_draws(draws, sizes, arm, field, length):
    numerator = np.zeros(length)
    denominator = np.zeros(length)
    for stratum, arms in sizes.items():
        if not arms[0] or not arms[1]:
            continue
        values = draws.get((stratum, arm, field))
        if values is None:
            continue
        valid = np.isfinite(values)
        weight = arms[0] + arms[1]
        numerator += np.where(valid, weight * values, 0.0)
        denominator += np.where(valid, weight, 0.0)
    result = np.full(length, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _interval(values, confidence, tail=None):
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None, 0
    tail = (1.0 - confidence) / 2.0 if tail is None else tail
    return [float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))], int(finite.size)


def _bounded_interval(lower, upper, confidence):
    """Interval for a partially identified effect.

    The lower endpoint and the upper endpoint are each given half the allowed
    error, so the union of the two one-sided intervals covers the true
    identification region with at least ``confidence`` (Bonferroni).  Reporting
    the raw bounds with a single sampling interval would understate uncertainty.
    """
    alpha = 1.0 - confidence
    lower_interval, lower_count = _interval(lower, confidence, tail=alpha / 2.0)
    upper_interval, upper_count = _interval(upper, confidence, tail=alpha / 2.0)
    if lower_interval is None or upper_interval is None:
        return None, 0
    return [lower_interval[0], upper_interval[1]], min(lower_count, upper_count)


def _scenario_status(defiers, compliers, control_attendance, invited_attendance):
    """Declared shares must be consistent with the attendance actually seen."""
    if defiers > control_attendance + _TOLERANCE:
        return "infeasible_defiers", (
            "declared defier share exceeds control-arm attendance, so such defiers "
            "cannot exist in these records")
    if compliers <= _TOLERANCE:
        return "no_compliers", "no complier share remains after the declared defier rate"
    if compliers > invited_attendance + _TOLERANCE:
        return "infeasible_compliers", (
            "declared complier share exceeds invited-arm attendance")
    return "supported", None


def _scenarios(config, weak, combined, itt_low, itt_high, scale, boot, confidence):
    """One row per declared defier/spillover combination.

    Write the observed invitation effect as

        ITT = N + C * theta + d * theta_defier

    where ``C`` is the complier share (first stage plus the declared defier
    rate), ``theta`` is the participation effect among compliers, and the defier
    contribution ``d * theta_defier`` lies inside ``d * [-spread, +spread]``
    because both potential outcomes lie on the outcome scale.  ``N`` is the part
    of the invitation effect that reaches people regardless of attendance; its
    declared magnitude is ``s * ITT`` with direction given by
    ``spillover_direction``.  Only the remaining ``(1 - s) * ITT`` is available
    to the complier ratio, giving the bounds below.

    The baseline row (``d = 0``, ``s = 0``) makes the smallest assumption and
    still reports a range whenever follow-up is missing or the first stage is
    imprecise.  No row is a data-identified fact.
    """
    control_attendance = combined[(0, "attendance")]
    invited_attendance = combined[(1, "attendance")]
    first_stage = None
    if control_attendance is not None and invited_attendance is not None:
        first_stage = invited_attendance - control_attendance
    low, high, _ = scale
    spread = None if low is None else high - low
    itt = None
    if None not in (combined[(0, "outcome")], combined[(1, "outcome")]):
        itt = combined[(1, "outcome")] - combined[(0, "outcome")]
    sign = 1.0 if config["spillover_direction"] == "same" else -1.0

    requested = {(config["defier_rate"], config["social_spillover"])}
    for defiers in config["defier_grid"]:
        for spillover in config["spillover_grid"]:
            requested.add((float(defiers), float(spillover)))

    rows = []
    for defiers, spillover in sorted(requested):
        row = {
            "defier_rate": float(defiers),
            "social_spillover": float(spillover),
            "spillover_direction": config["spillover_direction"],
            "direct_effect_component": None,
            "complier_share": None,
            "identification_bounds": None,
            "sampling_interval": None,
            "sampling_status": "not_defined",
            "sampling_replicates": 0,
            "status": "outcomes_unavailable",
            "reason": None,
        }
        if first_stage is None:
            row["reason"] = "attendance could not be summarised by invitation arm"
            rows.append(row)
            continue
        compliers = first_stage + defiers
        row["complier_share"] = float(compliers)
        status, reason = _scenario_status(defiers, compliers, control_attendance, invited_attendance)
        if status != "supported":
            row["status"], row["reason"] = status, reason
            rows.append(row)
            continue
        if itt_low is None or itt_high is None or spread is None or itt is None:
            row["reason"] = "no observed outcomes and no declared outcome scale"
            rows.append(row)
            continue
        # ``s`` is declared social share.  Only ``(1 - s)`` of the invitation
        # effect is available to the complier ratio; with ``opposite`` the
        # social part moves the other way, widening rather than shrinking.
        factor = (1.0 - spillover) if sign > 0 else (1.0 + spillover)
        direct = sign * spillover * itt
        row["direct_effect_component"] = float(direct)
        if abs(factor) <= _TOLERANCE:
            # Declaring that the whole invitation effect works through social
            # contact leaves no contrast for the complier ratio.  Reporting a
            # zero-width interval here would invent precision; say instead that
            # this scenario carries no information about participation.
            row["status"] = "no_residual_contrast"
            row["reason"] = ("the declared social share accounts for the whole invitation "
                             "effect, leaving no contrast for the complier ratio")
            rows.append(row)
            continue
        # Defiers are people who would attend only when *not* invited.  Their
        # effect can sit anywhere in the outcome spread, so worst cases add a
        # one-sided ``d * spread`` to each end.
        row["identification_bounds"] = [
            float((factor * itt_low - defiers * spread) / compliers),
            float((factor * itt_high + defiers * spread) / compliers),
        ]
        row["status"] = "supported"

        boot_first = boot["first_stage"]
        boot_low = boot["itt_low"]
        boot_high = boot["itt_high"]
        boot_control = boot["attendance_0"]
        boot_invited = boot["attendance_1"]
        if weak:
            row["sampling_status"] = "unbounded_weak_first_stage"
            row["reason"] = ("the first-stage interval reaches zero, so no finite sampling "
                             "interval for a participation effect is supportable")
            rows.append(row)
            continue
        denominator = boot_first + defiers
        valid = (np.isfinite(denominator) & (denominator > _TOLERANCE)
                 & np.isfinite(boot_low) & np.isfinite(boot_high))
        valid &= defiers <= boot_control + _TOLERANCE
        valid &= denominator <= boot_invited + _TOLERANCE
        valid_fraction = float(valid.mean()) if valid.size else 0.0
        if valid_fraction < MIN_VALID_RESAMPLES:
            row["sampling_status"] = "unstable"
            row["reason"] = (f"only {valid_fraction:.0%} of household resamples keep the "
                             "declared scenario feasible; no sampling interval is reported")
            rows.append(row)
            continue
        lower = np.where(valid, (factor * boot_low - defiers * spread) / denominator, np.nan)
        upper = np.where(valid, (factor * boot_high + defiers * spread) / denominator, np.nan)
        interval, count = _bounded_interval(lower, upper, confidence)
        if interval is None:
            row["sampling_status"] = "unstable"
            rows.append(row)
            continue
        row["sampling_interval"] = interval
        row["sampling_status"] = "bounded"
        row["sampling_replicates"] = count
        row["conclusion"] = _scenario_conclusion(row)
        rows.append(row)
    return rows


def _scenario_conclusion(row):
    """What one scenario alone can support, stated without inventing precision."""
    bounds = row.get("identification_bounds")
    if row["status"] != "supported" or bounds is None:
        return "unsupported"
    if bounds[0] > _TOLERANCE:
        return "positive"
    if bounds[1] < -_TOLERANCE:
        return "negative"
    return "inconclusive"


def _missing_arm(records, arm):
    selected = [row for row in records if row["invited"] == arm]
    members = sum(row["members"] for row in selected)
    missing = sum(row["missing"] for row in selected)
    return {"members": members, "missing": missing,
            "fraction": (missing / members) if members else None}


def _scenario_summary(scenarios, config):
    """Aggregate what the whole declared grid supports, without picking a winner.

    A conclusion is only called directional if every feasible scenario keeps the
    same sign.  If some scenarios are infeasible or unsupported, the honest
    summary says so rather than reporting only the most favourable row.
    """
    feasible = [row for row in scenarios if row["status"] == "supported"]
    infeasible = [row for row in scenarios if row["status"] != "supported"]
    low = high = None
    if feasible:
        lows = [row["identification_bounds"][0] for row in feasible]
        highs = [row["identification_bounds"][1] for row in feasible]
        low, high = float(min(lows)), float(max(highs))
    if not feasible:
        direction = "not_supported"
        detail = "no declared scenario leaves an identified complier share"
    elif low is not None and low > 0.0:
        direction = "positive_under_all_feasible_scenarios"
        detail = ("every feasible declared scenario keeps the participation effect above zero, "
                  "but the size still depends on the declared assumptions")
    elif high is not None and high < 0.0:
        direction = "negative_under_all_feasible_scenarios"
        detail = ("every feasible declared scenario keeps the participation effect below zero, "
                  "but the size still depends on the declared assumptions")
    else:
        direction = "sign_not_resolved"
        detail = ("feasible declared scenarios disagree on the sign, so no directional benefit "
                  "is claimed")
    return {
        "feasible_scenarios": len(feasible),
        "unsupported_scenarios": len(infeasible),
        "identification_envelope": None if low is None else [low, high],
        "direction": direction,
        "detail": detail,
        "baseline_scenario": {
            "defier_rate": config["defier_rate"],
            "social_spillover": config["social_spillover"],
        },
    }


def _diagnostics(source, invitation, participation, first_stage, weak, scale, scenarios):
    diagnostics = [{
        "code": "household_randomisation",
        "severity": "info",
        "detail": "Assignment is at household level; members are not independent observations.",
    }]
    missing = source["missing_followup"]
    if missing["total_missing"]:
        diagnostics.append({
            "code": "followup_missing",
            "severity": "caution",
            "detail": (f"{missing['total_missing']} of {source['people']} people have no "
                       "follow-up. Observed means exclude them and worst-case bounds use the "
                       "outcome scale; missing is not zero."),
        })
    else:
        diagnostics.append({
            "code": "followup_complete",
            "severity": "info",
            "detail": "Every person has a follow-up outcome.",
        })
    if invitation["status"] != "estimated":
        diagnostics.append({
            "code": "outcomes_unavailable",
            "severity": "blocking",
            "detail": "At least one arm has no follow-up outcome; outcome effects are unavailable.",
        })
    if first_stage is None:
        diagnostics.append({
            "code": "attendance_unavailable",
            "severity": "blocking",
            "detail": "Attendance could not be summarised by invitation arm.",
        })
    elif first_stage <= _TOLERANCE:
        diagnostics.append({
            "code": "weak_invitation",
            "severity": "blocking",
            "detail": ("Invitation did not increase attendance. A complier ratio would divide "
                       "by zero or flip sign, so no participation benefit is claimed."),
        })
    elif weak:
        diagnostics.append({
            "code": "weak_first_stage",
            "severity": "caution",
            "detail": ("The first-stage interval includes zero; sampling uncertainty for any "
                       "participation effect is unbounded, so only identification bounds are "
                       "reported."),
        })
    if source["excluded_households"]:
        diagnostics.append({
            "code": "excluded_households",
            "severity": "info",
            "detail": "Households without linked members are excluded from estimation.",
        })
    if source["strata_without_arm"]:
        diagnostics.append({
            "code": "stratum_without_arm",
            "severity": "caution",
            "detail": "Strata with an empty invitation arm contribute no contrast.",
        })
    if scale[2] == "unavailable":
        diagnostics.append({
            "code": "outcome_scale_unavailable",
            "severity": "blocking",
            "detail": ("No observed outcome and no declared outcome range; missing outcomes "
                       "cannot be bounded."),
        })
    if not any(row["status"] == "supported" for row in scenarios):
        diagnostics.append({
            "code": "no_supported_participation_scenario",
            "severity": "blocking",
            "detail": "No declared defier/spillover combination leaves a supported complier share.",
        })
    if participation["status"] == "partially_identified":
        diagnostics.append({
            "code": "participation_bounded_only",
            "severity": "caution",
            "detail": ("Participation effects are bounded, not point identified. The saved "
                       "scenario table records every assumption used."),
        })
    return diagnostics


def analyse(packet, options=None):
    config = parse_options(options)
    _check_design(packet)
    records = _household_records(packet)
    scale = _outcome_scale(packet, config["outcome_bounds"])
    prepared = _prepare(records, (scale[0], scale[1]))
    groups, means = _stratum_arm_means(prepared)
    households, members = _stratum_sizes(prepared)
    sizes = members
    strata_without_arm = sorted(stratum for stratum, arms in members.items()
                                if not arms[0] or not arms[1])

    combined = {(arm, field): _combine(means, sizes, arm, field)
                for arm in (0, 1)
                for field in RATIOS}

    first_stage = None
    if combined[(0, "attendance")] is not None and combined[(1, "attendance")] is not None:
        first_stage = combined[(1, "attendance")] - combined[(0, "attendance")]

    itt = None
    if combined[(0, "outcome")] is not None and combined[(1, "outcome")] is not None:
        itt = combined[(1, "outcome")] - combined[(0, "outcome")]

    # Missing follow-up bounds the invitation effect: the invited arm could be
    # as low as its lower bound while the control arm could be as high as its
    # upper bound, and vice versa.
    itt_low = itt_high = None
    if None not in (combined[(0, "outcome_high")], combined[(1, "outcome_low")]):
        itt_low = combined[(1, "outcome_low")] - combined[(0, "outcome_high")]
    if None not in (combined[(0, "outcome_low")], combined[(1, "outcome_high")]):
        itt_high = combined[(1, "outcome_high")] - combined[(0, "outcome_low")]

    raw_draws = _bootstrap_draws(groups, config["bootstrap_replicates"], config["seed"])
    boot = {}
    for arm in (0, 1):
        for field in RATIOS:
            boot[f"{field}_{arm}"] = _combine_draws(raw_draws, sizes, arm, field,
                                                    config["bootstrap_replicates"])
    boot["first_stage"] = boot["attendance_1"] - boot["attendance_0"]
    boot["itt"] = boot["outcome_1"] - boot["outcome_0"]
    boot["itt_low"] = boot["outcome_low_1"] - boot["outcome_high_0"]
    boot["itt_high"] = boot["outcome_high_1"] - boot["outcome_low_0"]

    first_stage_interval, _ = _interval(boot["first_stage"], config["confidence"])
    itt_interval, _ = _interval(boot["itt"], config["confidence"])
    weak = (first_stage is None or first_stage <= _TOLERANCE
            or first_stage_interval is None or first_stage_interval[0] <= 0.0)

    scenarios = _scenarios(config, weak, combined, itt_low, itt_high, scale, boot,
                           config["confidence"])
    baseline = next(row for row in scenarios
                    if abs(row["defier_rate"] - config["defier_rate"]) <= _TOLERANCE
                    and abs(row["social_spillover"] - config["social_spillover"]) <= _TOLERANCE)

    missing = {
        "invited_arm": _missing_arm(prepared, 1),
        "control_arm": _missing_arm(prepared, 0),
        "total_missing": sum(row["missing"] for row in prepared),
        "imputed_as_zero": False,
    }
    source = {
        "revision": packet["revision"],
        "households": len(records),
        "people": sum(row["members"] for row in records),
        "randomization": packet.get("randomization"),
        "strata": [{"stratum": stratum,
                    "control_households": households[stratum][0],
                    "invited_households": households[stratum][1]}
                   for stratum in sorted(sizes)],
        "excluded_households": [row["id"] for row in records if not row["members"]],
        "strata_without_arm": strata_without_arm,
        "missing_followup": missing,
    }

    invitation = {
        "estimand": "effect of being offered the course, regardless of attendance",
        "unit": "resident; households are the assignment and resampling unit",
        "status": "estimated" if itt is not None else "not_identifiable",
        "control_mean": combined[(0, "outcome")],
        "invited_mean": combined[(1, "outcome")],
        "estimate": itt,
        "interval": itt_interval,
        "confidence": config["confidence"],
        "missingness_bounds": None if itt_low is None else [float(itt_low), float(itt_high)],
        "reason": None if itt is not None else "no follow-up outcome in at least one arm",
    }

    baseline_supported = baseline["status"] == "supported"
    point_identified = (
        baseline_supported
        and baseline["identification_bounds"] is not None
        and abs(baseline["identification_bounds"][0]
                - baseline["identification_bounds"][1]) <= _TOLERANCE
        and missing["total_missing"] == 0
        and config["defier_rate"] == 0.0
        and config["social_spillover"] == 0.0
        and config["spillover_direction"] == "same"
        and not weak)
    if itt is None or first_stage is None:
        participation_status = "not_identified"
        participation_reason = "follow-up outcomes or attendance are unavailable"
    elif not baseline_supported:
        participation_status = "not_identified"
        participation_reason = ("the declared baseline scenario leaves no supported complier "
                                "share; no participation benefit is claimed")
    elif point_identified:
        participation_status = "identified_under_assumptions"
        participation_reason = None
    else:
        participation_status = "partially_identified"
        participation_reason = ("participation effects are bounded, not point identified; "
                                "missing follow-up, declared defier/spillover assumptions or "
                                "an imprecise first stage remain")

    participation = {
        "estimand": "effect of attending the course among households moved to attend by the invitation",
        "unit": "resident; households are the assignment and resampling unit",
        "status": participation_status,
        "first_stage": {
            "control_attendance": combined[(0, "attendance")],
            "invited_attendance": combined[(1, "attendance")],
            "estimate": first_stage,
            "interval": first_stage_interval,
            "confidence": config["confidence"],
            "compliers_if_no_defiers": first_stage,
            "defiers_feasible_max": combined[(0, "attendance")],
        },
        "outcome_scale": {
            "minimum": scale[0],
            "maximum": scale[1],
            "source": scale[2],
            "used_for_bounds": scale[0] is not None and missing["total_missing"] > 0,
        },
        "baseline": baseline,
        "scenarios": scenarios,
        "summary": _scenario_summary(scenarios, config),
        "reason": participation_reason,
    }

    assumptions = {
        "declared": {
            "defier_rate": config["defier_rate"],
            "defier_grid": config["defier_grid"],
            "social_spillover": config["social_spillover"],
            "spillover_grid": config["spillover_grid"],
            "spillover_direction": config["spillover_direction"],
            "outcome_bounds": config["outcome_bounds"],
            "confidence": config["confidence"],
        },
        "identifying": [
            "household-level random assignment with known stratum probabilities",
            "no interference between households",
            "invitation affects outcomes only through attendance plus the declared social share",
            "unobserved outcomes lie inside the declared or observed outcome range",
            "participation effects refer to households moved to attend by the invitation",
        ],
        "not_identified_by_data": [
            "the individual defier share",
            "the share of the invitation effect that works through social contact",
            "outcomes for people without follow-up",
        ],
    }

    diagnostics = _diagnostics(source, invitation, participation, first_stage, weak, scale, scenarios)

    return {
        "target": {
            "population": "households in the randomised invitation design",
            "assignment_unit": "household",
            "outcome_unit": "resident outcomes, kept inside their household during resampling",
            "replication_unit": "household",
            "co_residents": "share household assignment; never counted as independent draws",
            "coverage": "only people linked to randomised households",
        },
        "source": source,
        "invitation_effect": invitation,
        "participation_effect": participation,
        "assumptions": assumptions,
        "diagnostics": diagnostics,
        "precision": {
            "method": "stratified cluster bootstrap over households",
            "replicates": config["bootstrap_replicates"],
            "seed": config["seed"],
            "confidence": config["confidence"],
            "first_stage_interval": first_stage_interval,
            "invitation_interval": itt_interval,
            "weak_first_stage": weak,
        },
        "interpretation": _interpretation(invitation, participation),
    }


def _interpretation(invitation, participation):
    parts = []
    if invitation["status"] == "estimated":
        parts.append("The invitation effect is identified by household randomisation and is "
                     "reported separately from attendance.")
    else:
        parts.append("The invitation effect cannot be estimated from the observed follow-up.")
    if participation["status"] == "identified_under_assumptions":
        parts.append("The participation effect is point identified only under the declared "
                     "no-defier, no-spillover assumptions; its interval reflects household "
                     "sampling.")
    elif participation["status"] == "partially_identified":
        parts.append("The participation effect is bounded rather than point identified; the "
                     "scenario table shows how compliance, social spillover and defiers move "
                     "the supportable conclusion.")
    else:
        parts.append("The participation effect is not identified; no precise benefit figure is "
                     "reported and the uncertainty is retained.")
    return " ".join(parts)
