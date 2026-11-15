from collections import defaultdict
from app.protocol import table_mean


def household_view(packet):
    people = defaultdict(list)
    for person in packet["people"]:
        people[person["household"]].append(person)
    records = []
    for household in packet["households"]:
        members = people[household["household"]]
        records.append({**household, "size": len(members),
                        "baseline": table_mean([person["baseline"] for person in members]),
                        "participants": sum(person["attended"] for person in members),
                        "followups": sum(person["outcome"] is not None for person in members)})
    return records


def design_profile(packet):
    households = household_view(packet)
    strata = []
    for stratum in sorted({row["stratum"] for row in households}):
        groups = []
        for invited in (0, 1):
            rows = [row for row in households if row["stratum"] == stratum and row["invited"] == invited]
            weights = [1 / (row["invite_probability"] if invited else 1 - row["invite_probability"]) for row in rows]
            weight_sum = sum(weights)
            groups.append({"invited": invited, "households": len(rows),
                           "members": sum(row["size"] for row in rows),
                           "observed_followups": sum(row["followups"] for row in rows),
                           "household_baseline_mean": table_mean([row["baseline"] for row in rows]),
                           "weighted_households": weight_sum,
                           "effective_households": weight_sum ** 2 / sum(weight ** 2 for weight in weights) if weights else 0})
        strata.append({"stratum": stratum, "groups": groups})
    return {"randomization": packet["randomization"], "strata": strata,
            "empty_households": [row["household"] for row in households if not row["size"]],
            "weight_unit": "household"}
