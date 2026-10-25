"""
Core engine for Algorithm 1 ("Strategic Test-Time Compute Allocation") from
"Strategic Scaling of Test-Time Compute: A Bandit Learning Approach"
(arXiv:2506.12721).

Each query is a bandit arm; each "pull" generates `k` new responses for one
query *live* (via vLLM) and scores them *live* (via a reward oracle r(x, y)
in [0, 1] -- a PRM, a ground-truth verifier, or an LLM judge). A query is
eliminated from the active set as soon as a pulled response scores >= the
elimination threshold gamma. Generation, scoring, and elimination all
happen inside a single `pull_fn` call per round -- there is no separate
generate/score/pick pipeline; see eliminate.py for the dataset-specific
wiring.

Exploration rules (paper Section 4.3; i* denotes the highest-reward
response within a query's most-recently-pulled batch):

  uniform (base Algorithm 1): every round, pull k new responses for every
    active query.

  ucb / gap / entropy: every round, pick a single active query by a
    priority criterion, pull k for it only (finer-grained allocation).
    Let r_hat(x) = mean reward pulled so far, N(x) = #pulled so far:
      ucb:     argmax_x  r_hat(x) + lambda * N(x)^-0.5
      gap:     argmin_x  (gamma - r_hat(x)) * N(x)^-0.5
      entropy: argmax_x  H(x) + lambda * N(x)^-0.5
        H(x) = entropy of the distribution of *distinct* responses pulled
        so far for x (grouped by `entropy_key_fn`).
    N(x) == 0 is treated as maximal priority (pull every arm once first) --
    this tie-breaking isn't spelled out in the paper, it's the standard
    bandit convention.
"""

import math
from collections import Counter

import numpy as np


class QueryState:
    def __init__(self, query_id):
        self.id = query_id
        self.pulled = 0  # N(x)
        self.all_samples = []
        self.all_scores = []
        self.entropy_keys = []
        self.best_score = -math.inf
        self.best_sample = None
        self.eliminated = False

    def record(self, new_samples, new_scores, entropy_key_fn):
        self.pulled += len(new_samples)
        self.all_samples.extend(new_samples)
        self.all_scores.extend(new_scores)
        self.entropy_keys.extend(entropy_key_fn(s) for s in new_samples)

        i_star = int(np.argmax(new_scores))
        round_best_score = new_scores[i_star]
        if round_best_score > self.best_score:
            self.best_score = round_best_score
            self.best_sample = new_samples[i_star]
        return round_best_score


def _priority(state: QueryState, rule: str, gamma: float, ucb_lambda: float, entropy_lambda: float):
    n = state.pulled
    if n == 0:
        return math.inf if rule != "gap" else -math.inf

    r_hat = sum(state.all_scores) / n
    if rule == "ucb":
        return r_hat + ucb_lambda * n**-0.5
    if rule == "gap":
        return (gamma - r_hat) * n**-0.5
    if rule == "entropy":
        counts = Counter(state.entropy_keys)
        probs = np.array(list(counts.values()), dtype=float) / n
        h = float(-(probs * np.log(probs)).sum())
        return h + entropy_lambda * n**-0.5
    raise ValueError(f"Unknown exploration_rule: {rule}")


def run_algorithm1(
    items: list,
    pull_fn,
    total_budget: int,
    per_round_budget: int = 1,
    elimination_threshold: float = 1.0,
    exploration_rule: str = "uniform",
    ucb_lambda: float = 1.0,
    entropy_lambda: float = 3.0,
    max_samples: int | None = None,
    entropy_key_fn=lambda s: s.strip(),
    on_round=None,
) -> dict:
    """
    items: list of dicts, each with at least an "id" key; passed through to
    `pull_fn` unchanged.
    pull_fn(item, k) -> (samples: list[str], scores: list[float]): generates
    k new responses for `item` *and* scores them, live, in one call (e.g.
    one vLLM request + one PRM/verifier call per dataset-specific wiring in
    eliminate.py). May return fewer than k samples/scores if generation
    fails; an empty return eliminates the query (nothing left to pull).
    on_round(budget_remaining, active_count): optional progress callback.

    Returns {query_id: QueryState}.
    """
    items_by_id = {item["id"]: item for item in items}
    states = {qid: QueryState(qid) for qid in items_by_id}
    active = set(states.keys())
    budget = total_budget

    while budget > 0 and active:
        if exploration_rule == "uniform":
            round_ids = list(active)
        else:
            round_ids = [
                max(
                    active,
                    key=lambda qid: _priority(
                        states[qid], exploration_rule, elimination_threshold, ucb_lambda, entropy_lambda
                    ),
                )
            ]

        for qid in round_ids:
            if budget <= 0:
                break
            state = states[qid]

            k = per_round_budget
            if max_samples is not None:
                k = min(k, max_samples - state.pulled)
            k = min(k, budget)
            if k <= 0:
                active.discard(qid)
                continue

            new_samples, new_scores = pull_fn(items_by_id[qid], k)
            used = len(new_samples)
            budget -= used

            if used == 0:
                active.discard(qid)  # generation exhausted/failed
                continue

            round_best_score = state.record(new_samples, new_scores, entropy_key_fn)
            if round_best_score >= elimination_threshold:
                state.eliminated = True
                active.discard(qid)
            elif max_samples is not None and state.pulled >= max_samples:
                active.discard(qid)

        if on_round is not None:
            on_round(budget, len(active))

    return states
