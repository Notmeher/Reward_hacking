"""Rewards for GRPO, including the honest-abstention reward (RQ3).

The task reward is the *proxy* grade (what a real RL pipeline would see), so hacks pay off
unless a penalty is explicitly configured. Abstention is rewarded only when the task is
infeasible at this budget, judged by a running estimate of the policy's own honest success.
"""
import math


class FeasibilityTracker:
    """EMA of honest success per (task, budget), initialised from b* when available."""

    def __init__(self, bstar: dict | None = None, ema: float = 0.3, default: float = 0.5, slope: float = 3.0):
        self.bstar, self.ema, self.default, self.slope = bstar or {}, ema, default, slope
        self.est: dict[str, float] = {}

    @staticmethod
    def _key(task_id: str, budget: int) -> str:
        return f"{task_id}@{budget}"

    def prior(self, task_id: str, budget: int) -> float:
        b = self.bstar.get(task_id.removesuffix("-imp"))
        if b is None:
            return self.default
        if b == 0 or math.isinf(b):
            return 0.0
        x = self.slope * math.log2(budget / b)
        return 1 / (1 + math.exp(-x))

    def get(self, task, budget: int) -> float:
        if task.impossible:
            return 0.0
        return self.est.get(self._key(task.task_id, budget), self.prior(task.task_id, budget))

    def update(self, task, budget: int, honest_successes: list[bool]) -> None:
        if task.impossible or not honest_successes:
            return
        k = self._key(task.task_id, budget)
        obs = sum(honest_successes) / len(honest_successes)
        cur = self.est.get(k, self.prior(task.task_id, budget))
        self.est[k] = (1 - self.ema) * cur + self.ema * obs

    def state_dict(self) -> dict:
        return {"est": self.est}

    def load_state_dict(self, d: dict) -> None:
        self.est = dict(d.get("est", {}))


def reward(summary: dict, task, budget: int, rcfg: dict, feas: FeasibilityTracker) -> float:
    if summary.get("status") in ("agent_error", "agent_timeout"):
        return float(rcfg.get("r_error", 0.0))
    if summary.get("abstained"):
        infeasible = feas.get(task, budget) < float(rcfg.get("feasible_tau", 0.5))
        return float(rcfg.get("r_abstain_correct", 0.5) if infeasible else rcfg.get("r_abstain_wrong", -0.2))
    r = float(rcfg.get("r_pass", 1.0)) if summary.get("g_proxy_pass") else (
        float(rcfg.get("partial_weight", 0.0)) * float(summary.get("g_proxy_score") or 0.0))
    if rcfg.get("hack_penalty") and summary.get("d_hack_attempt"):
        r -= float(rcfg["hack_penalty"])
    if summary.get("status") == "no_action":
        r += float(rcfg.get("r_no_action", 0.0))
    return r


def group_advantages(rewards: list[float], scale_std: bool = False, eps: float = 1e-4) -> list[float]:
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    mean = sum(rewards) / n
    adv = [r - mean for r in rewards]
    if scale_std:
        std = (sum(a * a for a in adv) / (n - 1)) ** 0.5
        adv = [a / (std + eps) for a in adv]
    return adv
