from app.design import design_profile
from app.estimands import analyse as participation_analysis
from app.protocol import finite, table_mean, unique


def validate(packet):
    if not packet["revision"]:
        raise ValueError("missing revision")
    unique(packet["households"], "household")
    unique(packet["people"], "person")
    households = {row["household"]: row for row in packet["households"]}
    for row in packet["people"]:
        if row["household"] not in households or row["attended"] not in (0, 1):
            raise ValueError("unknown household or invalid participation")
        if row["outcome"] is not None:
            finite(row["outcome"], "outcome")
    for row in households.values():
        if row["invited"] not in (0, 1) or not 0 < row["invite_probability"] < 1:
            raise ValueError("invalid invitation design")


def invitation_balance(packet, options):
    households = {row["household"]: row for row in packet["households"]}
    groups = []
    for invited in (0, 1):
        selected = [row for row in packet["people"] if households[row["household"]]["invited"] == invited]
        groups.append({"invited": invited, "people": len(selected),
                       "households": len({row["household"] for row in selected}),
                       "attended": sum(row["attended"] for row in selected),
                       "observed_outcomes": sum(row["outcome"] is not None for row in selected),
                       "observed_mean": table_mean([row["outcome"] for row in selected])})
    return {"source_profile": design_profile(packet), "groups": groups, "interpretation": "Invitation balance and observed outcomes only; not a participation effect."}


def participation_effect(packet, options):
    return participation_analysis(packet, options)


ANALYSES = {"invitation_balance": invitation_balance,
            "participation_effect": participation_effect}
