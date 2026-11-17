"""Player Intelligence: opportunity-based poker statistics, uncertainty intervals, and profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Iterable

from .history import HandHistory, STREETS

_Z_95 = 1.959963984540054


def sample_adequacy_label(opportunities: int) -> str:
    if opportunities <= 0:
        return "no_observations"
    if opportunities < 20:
        return "insufficient_sample"
    if opportunities < 50:
        return "low_sample"
    if opportunities < 200:
        return "moderate_sample"
    return "adequate_sample"


def wilson_interval_95(successes: int, opportunities: int) -> tuple[float | None, float | None]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion."""
    if opportunities < 0 or successes < 0 or successes > opportunities:
        raise ValueError(f"invalid binomial counts: successes={successes}, opportunities={opportunities}")
    if opportunities == 0:
        return (None, None)
    n = float(opportunities)
    p_hat = successes / n
    z2 = _Z_95 * _Z_95
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    radicand = (p_hat * (1.0 - p_hat)) / n + (z2 / (4.0 * n * n))
    half_width = _Z_95 * math.sqrt(max(0.0, radicand)) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return (low, high)


@dataclass(frozen=True)
class RateEstimate:
    successes: int
    opportunities: int
    empirical_rate: float | None
    wilson_95_low: float | None
    wilson_95_high: float | None
    sample_adequacy: str
    bayesian_prior: str
    bayesian_mean: float
    bayesian_std: float

    @classmethod
    def from_counts(
        cls,
        successes: int,
        opportunities: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> "RateEstimate":
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        empirical = (successes / opportunities) if opportunities > 0 else None
        low, high = wilson_interval_95(successes, opportunities)
        alpha = prior_alpha + successes
        beta = prior_beta + (opportunities - successes)
        total = alpha + beta
        post_mean = alpha / total
        post_std = math.sqrt((alpha * beta) / (total * total * (total + 1.0)))
        return cls(
            successes=successes,
            opportunities=opportunities,
            empirical_rate=empirical,
            wilson_95_low=low,
            wilson_95_high=high,
            sample_adequacy=sample_adequacy_label(opportunities),
            bayesian_prior=f"Beta({prior_alpha:g}, {prior_beta:g})",
            bayesian_mean=post_mean,
            bayesian_std=post_std,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "opportunities": self.opportunities,
            "empirical_rate": self.empirical_rate,
            "wilson_95_ci": [self.wilson_95_low, self.wilson_95_high] if self.opportunities > 0 else None,
            "sample_adequacy": self.sample_adequacy,
            "bayesian_estimate": {
                "prior_assumption": self.bayesian_prior,
                "posterior_mean": self.bayesian_mean,
                "posterior_std": self.bayesian_std,
            },
        }


@dataclass(frozen=True)
class AggressionMetrics:
    bets: int
    raises: int
    calls: int
    checks: int
    folds: int
    aggression_frequency: RateEstimate
    aggression_factor: float | None

    @classmethod
    def from_counts(cls, bets: int, raises: int, calls: int, checks: int, folds: int) -> "AggressionMetrics":
        aggressive = bets + raises
        afq_denom = aggressive + calls + folds
        afq = RateEstimate.from_counts(aggressive, afq_denom)
        af = (aggressive / calls) if calls > 0 else None
        return cls(
            bets=bets,
            raises=raises,
            calls=calls,
            checks=checks,
            folds=folds,
            aggression_frequency=afq,
            aggression_factor=af,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bets": self.bets,
            "raises": self.raises,
            "calls": self.calls,
            "checks": self.checks,
            "folds": self.folds,
            "aggression_frequency": self.aggression_frequency.to_dict(),
            "aggression_factor": self.aggression_factor,
        }


@dataclass(frozen=True)
class BetSizingSummary:
    observations: int
    mean_pot_ratio: float | None
    median_pot_ratio: float | None
    min_pot_ratio: float | None
    max_pot_ratio: float | None
    small_bets_count: int
    medium_bets_count: int
    large_bets_count: int

    @classmethod
    def from_ratios(cls, ratios: list[float]) -> "BetSizingSummary":
        if not ratios:
            return cls(0, None, None, None, None, 0, 0, 0)
        small = sum(1 for r in ratios if r < 0.5)
        medium = sum(1 for r in ratios if 0.5 <= r <= 0.85)
        large = sum(1 for r in ratios if r > 0.85)
        return cls(
            observations=len(ratios),
            mean_pot_ratio=sum(ratios) / len(ratios),
            median_pot_ratio=float(median(ratios)),
            min_pot_ratio=min(ratios),
            max_pot_ratio=max(ratios),
            small_bets_count=small,
            medium_bets_count=medium,
            large_bets_count=large,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": self.observations,
            "mean_pot_ratio": self.mean_pot_ratio,
            "median_pot_ratio": self.median_pot_ratio,
            "min_pot_ratio": self.min_pot_ratio,
            "max_pot_ratio": self.max_pot_ratio,
            "buckets": {
                "small_lt_50pct": self.small_bets_count,
                "medium_50_to_85pct": self.medium_bets_count,
                "large_gt_85pct": self.large_bets_count,
            },
        }


@dataclass(frozen=True)
class PositionProfile:
    position: str
    hands: int
    vpip: RateEstimate
    pfr: RateEstimate
    three_bet: RateEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "hands": self.hands,
            "vpip": self.vpip.to_dict(),
            "pfr": self.pfr.to_dict(),
            "three_bet": self.three_bet.to_dict(),
        }


@dataclass(frozen=True)
class PlayerProfile:
    nickname: str
    hands_observed: int
    hand_ids: tuple[str, ...]
    vpip: RateEstimate
    pfr: RateEstimate
    three_bet: RateEstimate
    saw_flop: RateEstimate
    went_to_showdown: RateEstimate
    overall_aggression: AggressionMetrics
    postflop_aggression: AggressionMetrics
    by_position: dict[str, PositionProfile]
    bet_sizing_by_street: dict[str, BetSizingSummary]
    overall_bet_sizing: BetSizingSummary
    tendency_notes: tuple[str, ...]
    assumptions_and_limits: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "hands_observed": self.hands_observed,
            "hand_ids": list(self.hand_ids),
            "vpip": self.vpip.to_dict(),
            "pfr": self.pfr.to_dict(),
            "three_bet": self.three_bet.to_dict(),
            "saw_flop": self.saw_flop.to_dict(),
            "went_to_showdown": self.went_to_showdown.to_dict(),
            "overall_aggression": self.overall_aggression.to_dict(),
            "postflop_aggression": self.postflop_aggression.to_dict(),
            "by_position": {pos: prof.to_dict() for pos, prof in self.by_position.items()},
            "bet_sizing": {
                "overall": self.overall_bet_sizing.to_dict(),
                "by_street": {street: summary.to_dict() for street, summary in self.bet_sizing_by_street.items()},
            },
            "tendency_notes": list(self.tendency_notes),
            "assumptions_and_limits": list(self.assumptions_and_limits),
        }


def _build_tendency_notes(
    hands_observed: int,
    vpip: RateEstimate,
    pfr: RateEstimate,
    three_bet: RateEstimate,
    postflop_agg: AggressionMetrics,
) -> tuple[str, ...]:
    notes: list[str] = []
    if hands_observed < 20:
        notes.append(
            f"Very small sample ({hands_observed} hands observed): no player profile is assigned."
        )
        return tuple(notes)
    if hands_observed < 50:
        notes.append(
            f"Preliminary sample ({hands_observed} hands): 95% confidence intervals are still wide; avoid definitive labels."
        )

    if vpip.opportunities >= 20 and vpip.empirical_rate is not None:
        notes.append(
            f"Observed VPIP: {vpip.empirical_rate:.1%} (Wilson 95% CI: {vpip.wilson_95_low:.1%}–{vpip.wilson_95_high:.1%}, n={vpip.opportunities})."
        )
    if pfr.opportunities >= 20 and pfr.empirical_rate is not None:
        notes.append(
            f"Observed PFR: {pfr.empirical_rate:.1%} (Wilson 95% CI: {pfr.wilson_95_low:.1%}–{pfr.wilson_95_high:.1%}, n={pfr.opportunities})."
        )
    if three_bet.opportunities >= 20 and three_bet.empirical_rate is not None:
        notes.append(
            f"Observed 3-bet frequency: {three_bet.empirical_rate:.1%} (Wilson 95% CI: {three_bet.wilson_95_low:.1%}–{three_bet.wilson_95_high:.1%}, n={three_bet.opportunities})."
        )
    elif three_bet.opportunities > 0:
        notes.append(
            f"Only {three_bet.opportunities} observed 3-bet opportunities; too few to estimate a 3-bet frequency."
        )
    if postflop_agg.aggression_frequency.opportunities >= 20 and postflop_agg.aggression_frequency.empirical_rate is not None:
        afq = postflop_agg.aggression_frequency
        notes.append(
            f"Post-flop aggression frequency: {afq.empirical_rate:.1%} (95% CI: {afq.wilson_95_low:.1%}–{afq.wilson_95_high:.1%}, n={afq.opportunities})."
        )
    return tuple(notes)


def build_profiles(histories: Iterable[HandHistory]) -> dict[str, PlayerProfile]:
    """Aggregate deduplicated HandHistory records into per-nickname PlayerProfiles."""
    unique_histories: dict[str, HandHistory] = {}
    for history in histories:
        if history.hand_id not in unique_histories:
            unique_histories[history.hand_id] = history

    replays = {hand_id: hist.replay() for hand_id, hist in unique_histories.items()}
    nicknames: set[str] = set()
    for hist in unique_histories.values():
        for seat in hist.players:
            nicknames.add(seat.nickname)

    profiles: dict[str, PlayerProfile] = {}
    for nickname in sorted(nicknames):
        hand_ids: list[str] = []
        vpip_k = vpip_n = 0
        pfr_k = pfr_n = 0
        tb_k = tb_n = 0
        saw_flop_k = 0
        wtsd_k = 0

        bets = raises = calls = checks = folds = 0
        pf_bets = pf_raises = pf_calls = pf_checks = pf_folds = 0

        pos_counts: dict[str, dict[str, int]] = {}
        street_ratios: dict[str, list[float]] = {street: [] for street in STREETS}
        all_ratios: list[float] = []

        for hand_id, replay in replays.items():
            summary = replay.player_summaries.get(nickname)
            if summary is None:
                continue
            hand_ids.append(hand_id)
            if summary.vpip_opportunity:
                vpip_n += 1
                if summary.did_vpip:
                    vpip_k += 1
            if summary.pfr_opportunity:
                pfr_n += 1
                if summary.did_pfr:
                    pfr_k += 1
            if summary.three_bet_opportunity:
                tb_n += 1
                if summary.did_three_bet:
                    tb_k += 1
            if summary.saw_flop:
                saw_flop_k += 1
                if summary.went_to_showdown:
                    wtsd_k += 1

            bets += summary.bets
            raises += summary.raises
            calls += summary.calls
            checks += summary.checks
            folds += summary.folds

            pf_bets += summary.postflop_bets
            pf_raises += summary.postflop_raises
            pf_calls += summary.postflop_calls
            pf_checks += summary.postflop_checks
            pf_folds += summary.postflop_folds

            pos_bucket = pos_counts.setdefault(
                summary.position,
                {"hands": 0, "vpip_k": 0, "vpip_n": 0, "pfr_k": 0, "pfr_n": 0, "tb_k": 0, "tb_n": 0},
            )
            pos_bucket["hands"] += 1
            if summary.vpip_opportunity:
                pos_bucket["vpip_n"] += 1
                if summary.did_vpip:
                    pos_bucket["vpip_k"] += 1
            if summary.pfr_opportunity:
                pos_bucket["pfr_n"] += 1
                if summary.did_pfr:
                    pos_bucket["pfr_k"] += 1
            if summary.three_bet_opportunity:
                pos_bucket["tb_n"] += 1
                if summary.did_three_bet:
                    pos_bucket["tb_k"] += 1

            for action in replay.annotated_actions:
                if action.player == nickname and action.pot_ratio is not None:
                    street_ratios[action.street].append(action.pot_ratio)
                    all_ratios.append(action.pot_ratio)

        hands_observed = len(hand_ids)
        vpip_est = RateEstimate.from_counts(vpip_k, vpip_n)
        pfr_est = RateEstimate.from_counts(pfr_k, pfr_n)
        tb_est = RateEstimate.from_counts(tb_k, tb_n)
        saw_flop_est = RateEstimate.from_counts(saw_flop_k, hands_observed)
        wtsd_est = RateEstimate.from_counts(wtsd_k, saw_flop_k)

        overall_agg = AggressionMetrics.from_counts(bets, raises, calls, checks, folds)
        postflop_agg = AggressionMetrics.from_counts(pf_bets, pf_raises, pf_calls, pf_checks, pf_folds)

        by_position = {
            pos: PositionProfile(
                position=pos,
                hands=bucket["hands"],
                vpip=RateEstimate.from_counts(bucket["vpip_k"], bucket["vpip_n"]),
                pfr=RateEstimate.from_counts(bucket["pfr_k"], bucket["pfr_n"]),
                three_bet=RateEstimate.from_counts(bucket["tb_k"], bucket["tb_n"]),
            )
            for pos, bucket in sorted(pos_counts.items())
        }

        bet_sizing_by_street = {
            street: BetSizingSummary.from_ratios(ratios) for street, ratios in street_ratios.items()
        }
        overall_bet_sizing = BetSizingSummary.from_ratios(all_ratios)

        notes = _build_tendency_notes(hands_observed, vpip_est, pfr_est, tb_est, postflop_agg)
        assumptions = (
            "Empirical rates (k/n) and Wilson 95% confidence intervals are calculated directly from observed opportunities.",
            "The Bayesian estimate uses an explicit uniform Beta(1, 1) prior per metric and is shown separately from the empirical rate.",
            "Aggression frequency (AFq) = (bets + raises) / (bets + raises + calls + folds), excluding checks.",
            "Bet ratios measure the additional chips invested divided by the pot immediately before the action.",
            "Profiles are indexed only by the nickname visible at the table; real-world identity is not inferred.",
        )

        profiles[nickname] = PlayerProfile(
            nickname=nickname,
            hands_observed=hands_observed,
            hand_ids=tuple(hand_ids),
            vpip=vpip_est,
            pfr=pfr_est,
            three_bet=tb_est,
            saw_flop=saw_flop_est,
            went_to_showdown=wtsd_est,
            overall_aggression=overall_agg,
            postflop_aggression=postflop_agg,
            by_position=by_position,
            bet_sizing_by_street=bet_sizing_by_street,
            overall_bet_sizing=overall_bet_sizing,
            tendency_notes=notes,
            assumptions_and_limits=assumptions,
        )

    return profiles
