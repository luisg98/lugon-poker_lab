"""Small supervised frequency models distilled from validated 3-max charts.

Each model is intentionally narrow: equal stacks, one fixed payout structure,
no ante and preflop push/fold only. It fits one weighted linear frequency curve
per action and starting-hand class as a function of stack depth. It does not
replace the ICM solver outside that domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .preflop import HAND_CLASSES, combo_count, parse_hand_class

ACTION_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)


@dataclass(frozen=True)
class ThreeMaxFrequencyModel:
    """Per-class linear frequency curves over a bounded equal-stack domain."""

    payouts: tuple[float, float, float]
    stack_min_bb: float
    stack_max_bb: float
    stack_center_bb: float
    stack_scale_bb: float
    # action -> hand index -> (intercept, slope), on normalized stack.
    coefficients: tuple[tuple[tuple[float, float], ...], ...]
    training_stacks_bb: tuple[float, ...]
    training_examples: int
    validation: Mapping[str, float]
    training_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.payouts) != 3 or any(not math.isfinite(x) or x < 0.0 for x in self.payouts):
            raise ValueError("model payouts must contain three finite nonnegative values")
        if any(
            not math.isfinite(value)
            for value in (self.stack_min_bb, self.stack_max_bb, self.stack_center_bb)
        ) or self.stack_min_bb <= 0.0 or self.stack_max_bb <= self.stack_min_bb:
            raise ValueError("model stack range must be positive and nonempty")
        if self.stack_scale_bb <= 0.0 or not math.isfinite(self.stack_scale_bb):
            raise ValueError("model stack scale must be finite and positive")
        if len(self.coefficients) != len(ACTION_NAMES) or any(
            len(row) != len(HAND_CLASSES) for row in self.coefficients
        ):
            raise ValueError("model coefficients must contain six rows by 169 hand classes")
        if any(
            len(pair) != 2 or any(not math.isfinite(value) for value in pair)
            for row in self.coefficients
            for pair in row
        ):
            raise ValueError("model coefficients must be finite intercept/slope pairs")
        if self.training_examples <= 0 or not self.training_stacks_bb:
            raise ValueError("model must record nonempty training data")
        if any(not math.isfinite(value) for value in self.training_stacks_bb):
            raise ValueError("training stack depths must be finite")
        if any(not math.isfinite(value) for value in self.validation.values()):
            raise ValueError("validation metrics must be finite")
        if not isinstance(self.training_provenance, Mapping):
            raise ValueError("training provenance must be a mapping")

    def predict(
        self,
        hand_class: str,
        stack_bb: float,
        *,
        payouts: tuple[float, float, float] | None = None,
        ante_bb: float = 0.0,
        small_blind_bb: float = 0.5,
        big_blind_bb: float = 1.0,
    ) -> dict[str, float]:
        """Predict six action frequencies within the model's supported domain."""
        if payouts is not None and tuple(payouts) != self.payouts:
            raise ValueError("this model only supports its trained payout structure")
        if ante_bb != 0.0 or small_blind_bb != 0.5 or big_blind_bb != 1.0:
            raise ValueError("this model only supports 0 ante and blinds of 0.5/1 BB")
        if isinstance(stack_bb, bool) or not isinstance(stack_bb, (int, float)) or not math.isfinite(stack_bb):
            raise ValueError("stack_bb must be finite")
        stack = float(stack_bb)
        if stack < self.stack_min_bb or stack > self.stack_max_bb:
            raise ValueError(
                f"stack_bb must be within the trained range {self.stack_min_bb:g}–{self.stack_max_bb:g} BB"
            )
        index = HAND_CLASSES.index(parse_hand_class(hand_class))
        x = (stack - self.stack_center_bb) / self.stack_scale_bb
        return {
            action: min(1.0, max(0.0, intercept + slope * x))
            for action, row in zip(ACTION_NAMES, self.coefficients)
            for intercept, slope in (row[index],)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "lugon_three_max_frequency_model_v1",
            "model": "weighted_per_hand_linear_regression",
            "scope": {
                "players": 3,
                "equal_stacks_only": True,
                "payouts": list(self.payouts),
                "ante_bb": 0.0,
                "small_blind_bb": 0.5,
                "big_blind_bb": 1.0,
                "stack_range_bb": [self.stack_min_bb, self.stack_max_bb],
                "game_tree": "preflop_push_fold",
            },
            "features": ["169-way starting-hand class", "normalized equal stack depth"],
            "actions": {
                action: {
                    hand: list(self.coefficients[action_index][hand_index])
                    for hand_index, hand in enumerate(HAND_CLASSES)
                }
                for action_index, action in enumerate(ACTION_NAMES)
            },
            "normalization": {
                "stack_center_bb": self.stack_center_bb,
                "stack_scale_bb": self.stack_scale_bb,
                "formula": "x = (stack_bb - stack_center_bb) / stack_scale_bb",
            },
            "training": {
                "examples": self.training_examples,
                "stack_nodes_bb": list(self.training_stacks_bb),
                "label_source": "mean of stratified solver charts across seeds",
                "weighting": "inverse variance from seed disagreement, with standard deviation floored at 0.05",
                "provenance": dict(self.training_provenance),
            },
            "leave_one_stack_out_validation": dict(self.validation),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ThreeMaxFrequencyModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "lugon_three_max_frequency_model_v1":
            raise ValueError("unsupported three-max model format")
        scope = payload["scope"]
        if (
            scope.get("players") != 3
            or scope.get("equal_stacks_only") is not True
            or float(scope.get("ante_bb", -1.0)) != 0.0
            or float(scope.get("small_blind_bb", -1.0)) != 0.5
            or float(scope.get("big_blind_bb", -1.0)) != 1.0
            or scope.get("game_tree") != "preflop_push_fold"
        ):
            raise ValueError("model file has an unsupported game scope")
        normalization = payload["normalization"]
        actions = payload["actions"]
        coefficients = tuple(
            tuple(tuple(float(value) for value in actions[action][hand]) for hand in HAND_CLASSES)
            for action in ACTION_NAMES
        )
        training = payload["training"]
        training_provenance = training.get("provenance", {})
        if not isinstance(training_provenance, Mapping):
            raise ValueError("model training provenance must be an object")
        return cls(
            payouts=tuple(float(value) for value in scope["payouts"]),
            stack_min_bb=float(scope["stack_range_bb"][0]),
            stack_max_bb=float(scope["stack_range_bb"][1]),
            stack_center_bb=float(normalization["stack_center_bb"]),
            stack_scale_bb=float(normalization["stack_scale_bb"]),
            coefficients=coefficients,
            training_stacks_bb=tuple(float(value) for value in training["stack_nodes_bb"]),
            training_examples=int(training["examples"]),
            validation={key: float(value) for key, value in payload["leave_one_stack_out_validation"].items()},
            training_provenance=dict(training_provenance),
        )


@dataclass(frozen=True)
class UnequalStackThreeMaxFrequencyModel:
    """Experimental frequency model with one feature per seat's stack depth."""

    payouts: tuple[float, float, float]
    stack_min_by_seat_bb: tuple[float, float, float]
    stack_max_by_seat_bb: tuple[float, float, float]
    stack_center_by_seat_bb: tuple[float, float, float]
    stack_scale_by_seat_bb: tuple[float, float, float]
    # action -> hand index -> intercept and normalized stack feature coefficients.
    coefficients: tuple[tuple[tuple[float, ...], ...], ...]
    training_stack_vectors_bb: tuple[tuple[float, float, float], ...]
    training_examples: int
    validation: Mapping[str, float]
    training_provenance: Mapping[str, Any] = field(default_factory=dict)
    interaction_terms: bool = False

    def __post_init__(self) -> None:
        if len(self.payouts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in self.payouts):
            raise ValueError("model payouts must contain three finite nonnegative values")
        if sum(self.payouts) <= 0.0 or any(
            self.payouts[index] < self.payouts[index + 1] for index in range(2)
        ):
            raise ValueError("model payouts must be nonzero and ordered")
        for values, label in (
            (self.stack_min_by_seat_bb, "stack minima"),
            (self.stack_max_by_seat_bb, "stack maxima"),
            (self.stack_center_by_seat_bb, "stack centers"),
            (self.stack_scale_by_seat_bb, "stack scales"),
        ):
            if len(values) != 3 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"model {label} must contain three finite values")
        if any(
            minimum <= 0.0 or maximum <= minimum or scale <= 0.0
            for minimum, maximum, scale in zip(
                self.stack_min_by_seat_bb,
                self.stack_max_by_seat_bb,
                self.stack_scale_by_seat_bb,
            )
        ):
            raise ValueError("model stack ranges must be positive and nonempty")
        if len(self.coefficients) != len(ACTION_NAMES) or any(
            len(row) != len(HAND_CLASSES) for row in self.coefficients
        ):
            raise ValueError("model coefficients must contain six rows by 169 hand classes")
        expected_coefficients = 7 if self.interaction_terms else 4
        if any(
            len(values) != expected_coefficients or any(not math.isfinite(value) for value in values)
            for row in self.coefficients
            for values in row
        ):
            raise ValueError("model coefficients do not match the selected stack feature set")
        if self.training_examples <= 0 or len(self.training_stack_vectors_bb) < 5:
            raise ValueError("model must record at least five training stack vectors")
        if any(
            len(vector) != 3 or any(not math.isfinite(value) for value in vector)
            for vector in self.training_stack_vectors_bb
        ):
            raise ValueError("training stack vectors must contain three finite values")
        if any(not math.isfinite(value) for value in self.validation.values()):
            raise ValueError("validation metrics must be finite")
        if not isinstance(self.training_provenance, Mapping):
            raise ValueError("training provenance must be a mapping")

    def predict(
        self,
        hand_class: str,
        stacks_bb: Sequence[float],
        *,
        payouts: tuple[float, float, float] | None = None,
        ante_bb: float = 0.0,
        small_blind_bb: float = 0.5,
        big_blind_bb: float = 1.0,
    ) -> dict[str, float]:
        """Predict six frequencies for a BTN/SB/BB stack vector in-domain."""
        if payouts is not None and tuple(payouts) != self.payouts:
            raise ValueError("this model only supports its trained payout structure")
        if ante_bb != 0.0 or small_blind_bb != 0.5 or big_blind_bb != 1.0:
            raise ValueError("this model only supports 0 ante and blinds of 0.5/1 BB")
        if isinstance(stacks_bb, (str, bytes)) or len(stacks_bb) != 3:
            raise ValueError("stacks_bb must contain BTN, SB and BB stack depths")
        stacks: list[float] = []
        for value in stacks_bb:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("stack depths must be finite numbers")
            stacks.append(float(value))
        if any(
            value < minimum or value > maximum
            for value, minimum, maximum in zip(
                stacks,
                self.stack_min_by_seat_bb,
                self.stack_max_by_seat_bb,
            )
        ):
            raise ValueError("a stack depth lies outside the trained seat-specific range")
        normalized = tuple(
            (value - center) / scale
            for value, center, scale in zip(
                stacks,
                self.stack_center_by_seat_bb,
                self.stack_scale_by_seat_bb,
            )
        )
        features = normalized
        if self.interaction_terms:
            btn, sb, bb = normalized
            features = (*normalized, btn * sb, btn * bb, sb * bb)
        index = HAND_CLASSES.index(parse_hand_class(hand_class))
        return {
            action: min(
                1.0,
                max(0.0, intercept + sum(slope * feature for slope, feature in zip(slopes, features))),
            )
            for action, row in zip(ACTION_NAMES, self.coefficients)
            for intercept, *slopes in (row[index],)
        }

    def to_dict(self) -> dict[str, Any]:
        model_format = (
            "lugon_three_max_unequal_stack_frequency_model_v2"
            if self.interaction_terms
            else "lugon_three_max_unequal_stack_frequency_model_v1"
        )
        return {
            "format": model_format,
            "model": (
                "weighted_per_hand_regression_with_pairwise_stack_interactions"
                if self.interaction_terms
                else "weighted_per_hand_linear_regression_three_stack_features"
            ),
            "scope": {
                "players": 3,
                "equal_stacks_only": False,
                "payouts": list(self.payouts),
                "ante_bb": 0.0,
                "small_blind_bb": 0.5,
                "big_blind_bb": 1.0,
                "stack_range_by_seat_bb": {
                    seat: [self.stack_min_by_seat_bb[index], self.stack_max_by_seat_bb[index]]
                    for index, seat in enumerate(("BTN", "SB", "BB"))
                },
                "game_tree": "preflop_push_fold",
            },
            "features": [
                "169-way starting-hand class",
                "BTN, SB and BB stack depths",
                *(["BTN×SB", "BTN×BB", "SB×BB"] if self.interaction_terms else []),
            ],
            "actions": {
                action: {
                    hand: list(self.coefficients[action_index][hand_index])
                    for hand_index, hand in enumerate(HAND_CLASSES)
                }
                for action_index, action in enumerate(ACTION_NAMES)
            },
            "normalization": {
                "stack_center_by_seat_bb": dict(zip(("BTN", "SB", "BB"), self.stack_center_by_seat_bb)),
                "stack_scale_by_seat_bb": dict(zip(("BTN", "SB", "BB"), self.stack_scale_by_seat_bb)),
                "formula": "x_seat = (stack_seat_bb - center_seat_bb) / scale_seat_bb",
            },
            "training": {
                "examples": self.training_examples,
                "stack_vectors_bb": [list(vector) for vector in self.training_stack_vectors_bb],
                "label_source": "mean of stratified solver charts across seeds",
                "weighting": "inverse variance from seed disagreement, with standard deviation floored at 0.05",
                "provenance": dict(self.training_provenance),
            },
            "leave_one_stack_vector_out_validation": dict(self.validation),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "UnequalStackThreeMaxFrequencyModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_format = payload.get("format")
        if model_format not in (
            "lugon_three_max_unequal_stack_frequency_model_v1",
            "lugon_three_max_unequal_stack_frequency_model_v2",
        ):
            raise ValueError("unsupported unequal-stack three-max model format")
        interaction_terms = model_format == "lugon_three_max_unequal_stack_frequency_model_v2"
        scope = payload["scope"]
        if (
            scope.get("players") != 3
            or scope.get("equal_stacks_only") is not False
            or float(scope.get("ante_bb", -1.0)) != 0.0
            or float(scope.get("small_blind_bb", -1.0)) != 0.5
            or float(scope.get("big_blind_bb", -1.0)) != 1.0
            or scope.get("game_tree") != "preflop_push_fold"
        ):
            raise ValueError("model file has an unsupported game scope")
        normalization = payload["normalization"]
        actions = payload["actions"]
        seats = ("BTN", "SB", "BB")
        coefficients = tuple(
            tuple(tuple(float(value) for value in actions[action][hand]) for hand in HAND_CLASSES)
            for action in ACTION_NAMES
        )
        training = payload["training"]
        provenance = training.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("model training provenance must be an object")
        ranges = scope["stack_range_by_seat_bb"]
        return cls(
            payouts=tuple(float(value) for value in scope["payouts"]),
            stack_min_by_seat_bb=tuple(float(ranges[seat][0]) for seat in seats),
            stack_max_by_seat_bb=tuple(float(ranges[seat][1]) for seat in seats),
            stack_center_by_seat_bb=tuple(float(normalization["stack_center_by_seat_bb"][seat]) for seat in seats),
            stack_scale_by_seat_bb=tuple(float(normalization["stack_scale_by_seat_bb"][seat]) for seat in seats),
            coefficients=coefficients,
            training_stack_vectors_bb=tuple(
                tuple(float(value) for value in vector) for vector in training["stack_vectors_bb"]
            ),
            training_examples=int(training["examples"]),
            validation={
                key: float(value)
                for key, value in payload["leave_one_stack_vector_out_validation"].items()
            },
            training_provenance=dict(provenance),
            interaction_terms=interaction_terms,
        )


def _fit_weighted_line(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Fit y=a+b*x with inverse-variance weights; points are x,y,label_sd."""
    if not points:
        raise ValueError("cannot fit an empty frequency curve")
    weights = [1.0 / max(label_sd, 0.05) ** 2 for _, _, label_sd in points]
    total_weight = sum(weights)
    mean_x = sum(weight * x for weight, (x, _, _) in zip(weights, points)) / total_weight
    mean_y = sum(weight * y for weight, (_, y, _) in zip(weights, points)) / total_weight
    variance_x = sum(weight * (x - mean_x) ** 2 for weight, (x, _, _) in zip(weights, points))
    if variance_x <= 1e-15:
        return mean_y, 0.0
    covariance = sum(
        weight * (x - mean_x) * (y - mean_y)
        for weight, (x, y, _) in zip(weights, points)
    )
    slope = covariance / variance_x
    return mean_y - slope * mean_x, slope


def _fit_weighted_stack_surface(
    points: list[tuple[tuple[float, float, float], float, float]],
    *,
    interaction_terms: bool = False,
) -> tuple[float, ...]:
    """Fit weighted least squares with optional pairwise stack interactions."""
    dimension = 7 if interaction_terms else 4
    if len(points) < dimension:
        raise ValueError(f"at least {dimension} distinct stack vectors are required")
    normal = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    for features, frequency, label_sd in points:
        if len(features) != 3:
            raise ValueError("stack surface features must contain BTN, SB and BB")
        expanded = features
        if interaction_terms:
            btn, sb, bb = features
            expanded = (*features, btn * sb, btn * bb, sb * bb)
        row = (1.0, *expanded)
        weight = 1.0 / max(label_sd, 0.05) ** 2
        for i in range(dimension):
            target[i] += weight * row[i] * frequency
            for j in range(dimension):
                normal[i][j] += weight * row[i] * row[j]

    augmented = [normal[row] + [target[row]] for row in range(dimension)]
    for column in range(dimension):
        pivot = max(range(column, dimension), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ValueError("training stack vectors do not span all three seat features")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    coefficients = tuple(augmented[row][-1] for row in range(dimension))
    if any(not math.isfinite(value) for value in coefficients):
        raise ValueError("stack surface fit produced non-finite coefficients")
    return coefficients


def train_three_max_frequency_model(
    rows: Iterable[Mapping[str, Any]],
    *,
    payouts: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> ThreeMaxFrequencyModel:
    """Fit a small model from validated JSONL solver labels.

    The trainer refuses mixed payouts, unequal stacks, antes, or nonstandard
    blinds. Seed disagreement is retained in the labels and used as a fitting
    weight; it is not silently discarded.
    """
    if len(payouts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in payouts):
        raise ValueError("payouts must contain three finite nonnegative values")
    if sum(payouts) <= 0.0 or any(payouts[index] < payouts[index + 1] for index in range(2)):
        raise ValueError("payouts must be nonzero and ordered")
    examples = list(rows)
    if not examples:
        raise ValueError("training data is empty")

    grouped: dict[str, dict[int, list[tuple[float, float, float]]]] = {
        action: {index: [] for index in range(len(HAND_CLASSES))} for action in ACTION_NAMES
    }
    training_stacks: set[float] = set()
    seen_examples: set[tuple[int, float]] = set()
    shared_training_provenance: dict[str, Any] | None = None
    for row in examples:
        if row.get("format") != "lugon_3max_push_fold_label_v1":
            raise ValueError("training row has an unsupported label format")
        features = row["features"]
        row_payouts = tuple(float(value) for value in features["payouts"])
        if row_payouts != tuple(payouts):
            raise ValueError("dataset contains a different payout structure")
        stacks = features["stacks_bb"]
        seat_stacks = tuple(float(stacks[seat]) for seat in ("BTN", "SB", "BB"))
        if any(not math.isfinite(value) or value <= 0.0 for value in seat_stacks):
            raise ValueError("stack depths must be finite and positive")
        if not (seat_stacks[0] == seat_stacks[1] == seat_stacks[2]):
            raise ValueError("this baseline trainer only accepts equal stacks")
        if float(features.get("ante_bb_per_player", 0.0)) != 0.0:
            raise ValueError("this baseline trainer only accepts zero ante")
        if float(features.get("small_blind_bb", 0.5)) != 0.5 or float(features.get("big_blind_bb", 1.0)) != 1.0:
            raise ValueError("this baseline trainer requires 0.5/1 BB blinds")
        provenance = row.get("label_provenance", {})
        if provenance.get("sampling_method") != "stratified_by_seat_and_hand_class":
            raise ValueError("training labels must come from the stratified solver")
        if provenance.get("all_seeds_within_tolerance") is not True:
            raise ValueError("every solver seed must meet the training convergence tolerance")
        independent_validation = provenance.get("independent_stratified_validation")
        if not isinstance(independent_validation, Mapping):
            raise ValueError("every training example must have independent validation provenance")
        if independent_validation.get("method") != "stratified_crossfit":
            raise ValueError("training labels need independent stratified cross-fit validation")
        solver = provenance.get("solver")
        if not isinstance(solver, str) or not solver:
            raise ValueError("training labels must identify the solver that produced them")
        sample_count = provenance.get("samples_per_hand_class")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("training labels must record a positive sample count per hand class")
        raw_seeds = provenance.get("seeds")
        if not isinstance(raw_seeds, list) or len(raw_seeds) < 2:
            raise ValueError("training labels must record at least two solver seeds")
        solver_seeds: list[int] = []
        for seed in raw_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("solver seeds must be nonnegative integers")
            solver_seeds.append(seed)
        if len(set(solver_seeds)) != len(solver_seeds):
            raise ValueError("solver seeds must be unique")
        deals_per_seed = provenance.get("deals_per_seed")
        expected_deals = 3 * len(HAND_CLASSES) * sample_count
        if isinstance(deals_per_seed, bool) or not isinstance(deals_per_seed, int) or deals_per_seed != expected_deals:
            raise ValueError("training deal count does not match the recorded stratified sample count")
        validation_samples = independent_validation.get("samples_per_seat_hand_class")
        if isinstance(validation_samples, bool) or not isinstance(validation_samples, int) or validation_samples < 2:
            raise ValueError("independent validation must use at least two deals per seat and hand class")
        validation_seed = independent_validation.get("seed")
        if isinstance(validation_seed, bool) or not isinstance(validation_seed, int) or validation_seed < 0:
            raise ValueError("independent validation must record a nonnegative seed")
        validation_deals = independent_validation.get("total_physical_deals")
        if (
            isinstance(validation_deals, bool)
            or not isinstance(validation_deals, int)
            or validation_deals != 3 * len(HAND_CLASSES) * validation_samples
        ):
            raise ValueError("validation deal count does not match the recorded stratified sample count")
        current_provenance = {
            "solver": solver,
            "sampling_method": provenance.get("sampling_method"),
            "solver_seeds": solver_seeds,
            "samples_per_hand_class_per_seed": sample_count,
            "independent_validation": {
                "method": independent_validation["method"],
                "samples_per_seat_hand_class": validation_samples,
                "seed": validation_seed,
            },
        }
        if shared_training_provenance is None:
            shared_training_provenance = current_provenance
        elif current_provenance != shared_training_provenance:
            raise ValueError("training labels mix solver seeds or sampling/validation configurations")
        validation_gap = float(independent_validation["mean_strategy_plugin_nash_conv"])
        validation_crossfit = float(independent_validation["mean_strategy_crossfit_br_gain"])
        if not math.isfinite(validation_gap) or validation_gap < 0.0 or not math.isfinite(validation_crossfit):
            raise ValueError("independent validation metrics must be finite and NashConv nonnegative")
        hand = parse_hand_class(str(features["hand_class"]))
        hand_index = HAND_CLASSES.index(hand)
        stack = seat_stacks[0]
        example_key = (hand_index, stack)
        if example_key in seen_examples:
            raise ValueError(f"duplicate label for hand class {hand} at {stack:g} BB")
        seen_examples.add(example_key)
        training_stacks.add(stack)
        targets = row["targets"]
        for action in ACTION_NAMES:
            target = targets[action]
            frequency = float(target["mean_frequency"])
            label_sd = float(target["seed_standard_deviation"])
            if not math.isfinite(frequency) or not 0.0 <= frequency <= 1.0:
                raise ValueError("action frequencies must be finite values in [0, 1]")
            if not math.isfinite(label_sd) or label_sd < 0.0:
                raise ValueError("label seed deviations must be finite and nonnegative")
            grouped[action][hand_index].append((stack, frequency, label_sd))

    stack_nodes = tuple(sorted(training_stacks))
    if len(stack_nodes) < 3:
        raise ValueError("at least three distinct stack depths are required")
    stack_min, stack_max = stack_nodes[0], stack_nodes[-1]
    center = (stack_min + stack_max) / 2.0
    scale = (stack_max - stack_min) / 2.0

    def normalized_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        return [((stack - center) / scale, frequency, label_sd) for stack, frequency, label_sd in points]

    coefficients: list[tuple[tuple[float, float], ...]] = []
    errors: list[tuple[float, float]] = []
    baseline_errors: list[tuple[float, int]] = []
    total_label_sd: list[tuple[float, int]] = []
    errors_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    baseline_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    errors_by_stack: dict[float, list[tuple[float, int]]] = {stack: [] for stack in stack_nodes}

    for action in ACTION_NAMES:
        action_coefficients: list[tuple[float, float]] = []
        for hand_index, hand in enumerate(HAND_CLASSES):
            points = grouped[action][hand_index]
            by_stack = {stack: (frequency, label_sd) for stack, frequency, label_sd in points}
            if set(by_stack) != set(stack_nodes):
                raise ValueError(f"hand class {hand} is missing a training stack for {action}")
            normalized = normalized_points(points)
            intercept, slope = _fit_weighted_line(normalized)
            action_coefficients.append((intercept, slope))

            for held_stack in stack_nodes:
                train_points = normalized_points([point for point in points if point[0] != held_stack])
                test_frequency, _ = by_stack[held_stack]
                loo_intercept, loo_slope = _fit_weighted_line(train_points)
                test_x = (held_stack - center) / scale
                prediction = min(1.0, max(0.0, loo_intercept + loo_slope * test_x))
                error = abs(prediction - test_frequency)
                combo_weight = combo_count(hand)
                errors.append((error, combo_weight))
                errors_by_action[action].append((error, combo_weight))
                errors_by_stack[held_stack].append((error, combo_weight))
                baseline_points = [frequency for stack, frequency, _ in points if stack != held_stack]
                baseline = sum(baseline_points) / len(baseline_points)
                baseline_error = abs(baseline - test_frequency)
                baseline_errors.append((baseline_error, combo_weight))
                baseline_by_action[action].append((baseline_error, combo_weight))
                total_label_sd.append((by_stack[held_stack][1], combo_weight))

        coefficients.append(tuple(action_coefficients))

    weight_sum = sum(weight for _, weight in errors)
    model_mae = sum(error * weight for error, weight in errors) / weight_sum
    model_rmse = math.sqrt(sum(error * error * weight for error, weight in errors) / weight_sum)
    baseline_mae = sum(error * weight for error, weight in baseline_errors) / sum(
        weight for _, weight in baseline_errors
    )
    mean_label_sd = sum(value * weight for value, weight in total_label_sd) / sum(
        weight for _, weight in total_label_sd
    )
    validation = {
        "leave_one_stack_out_combo_weighted_mae": model_mae,
        "leave_one_stack_out_combo_weighted_rmse": model_rmse,
        "constant_per_hand_baseline_combo_weighted_mae": baseline_mae,
        "mae_reduction_vs_constant_baseline": baseline_mae - model_mae,
        "mean_seed_label_standard_deviation": mean_label_sd,
    }
    for action in ACTION_NAMES:
        action_weight = sum(weight for _, weight in errors_by_action[action])
        validation[f"loo_mae_{action}"] = sum(
            error * weight for error, weight in errors_by_action[action]
        ) / action_weight
        validation[f"baseline_mae_{action}"] = sum(
            error * weight for error, weight in baseline_by_action[action]
        ) / action_weight
    for stack, stack_errors in errors_by_stack.items():
        stack_weight = sum(weight for _, weight in stack_errors)
        validation[f"loo_mae_at_{stack:g}bb"] = sum(
            error * weight for error, weight in stack_errors
        ) / stack_weight
    return ThreeMaxFrequencyModel(
        payouts=tuple(float(value) for value in payouts),
        stack_min_bb=stack_min,
        stack_max_bb=stack_max,
        stack_center_bb=center,
        stack_scale_bb=scale,
        coefficients=tuple(coefficients),
        training_stacks_bb=stack_nodes,
        training_examples=len(examples),
        validation=validation,
        training_provenance=shared_training_provenance or {},
    )


def train_unequal_stack_three_max_frequency_model(
    rows: Iterable[Mapping[str, Any]],
    *,
    payouts: tuple[float, float, float] = (0.5, 0.3, 0.2),
    interaction_terms: bool = False,
) -> UnequalStackThreeMaxFrequencyModel:
    """Fit a weighted stack-feature model from validated unequal-stack labels."""
    if len(payouts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in payouts):
        raise ValueError("payouts must contain three finite nonnegative values")
    if sum(payouts) <= 0.0 or any(payouts[index] < payouts[index + 1] for index in range(2)):
        raise ValueError("payouts must be nonzero and ordered")
    examples = list(rows)
    if not examples:
        raise ValueError("training data is empty")

    grouped: dict[str, dict[int, list[tuple[tuple[float, float, float], float, float]]]] = {
        action: {index: [] for index in range(len(HAND_CLASSES))} for action in ACTION_NAMES
    }
    stack_vectors: set[tuple[float, float, float]] = set()
    seen_examples: set[tuple[int, tuple[float, float, float]]] = set()
    shared_provenance: dict[str, Any] | None = None

    for row in examples:
        if row.get("format") != "lugon_3max_push_fold_label_v1":
            raise ValueError("training row has an unsupported label format")
        features = row["features"]
        row_payouts = tuple(float(value) for value in features["payouts"])
        if row_payouts != tuple(payouts):
            raise ValueError("dataset contains a different payout structure")
        raw_stacks = features["stacks_bb"]
        stack_vector = tuple(float(raw_stacks[seat]) for seat in ("BTN", "SB", "BB"))
        if any(not math.isfinite(value) or value <= 0.0 for value in stack_vector):
            raise ValueError("stack depths must be finite and positive")
        if float(features.get("ante_bb_per_player", 0.0)) != 0.0:
            raise ValueError("this experimental model only accepts zero ante")
        if (
            float(features.get("small_blind_bb", 0.5)) != 0.5
            or float(features.get("big_blind_bb", 1.0)) != 1.0
        ):
            raise ValueError("this experimental model requires 0.5/1 BB blinds")

        provenance = row.get("label_provenance", {})
        if provenance.get("sampling_method") != "stratified_by_seat_and_hand_class":
            raise ValueError("training labels must come from the stratified solver")
        if provenance.get("all_seeds_within_tolerance") is not True:
            raise ValueError("every solver seed must meet the training convergence tolerance")
        solver = provenance.get("solver")
        if not isinstance(solver, str) or not solver:
            raise ValueError("training labels must identify the solver that produced them")
        raw_seeds = provenance.get("seeds")
        if not isinstance(raw_seeds, list) or len(raw_seeds) < 2:
            raise ValueError("training labels must record at least two solver seeds")
        solver_seeds: list[int] = []
        for seed in raw_seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("solver seeds must be nonnegative integers")
            solver_seeds.append(seed)
        if len(set(solver_seeds)) != len(solver_seeds):
            raise ValueError("solver seeds must be unique")
        sample_count = provenance.get("samples_per_hand_class")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("training labels must record a positive sample count per hand class")
        deals_per_seed = provenance.get("deals_per_seed")
        if (
            isinstance(deals_per_seed, bool)
            or not isinstance(deals_per_seed, int)
            or deals_per_seed != 3 * len(HAND_CLASSES) * sample_count
        ):
            raise ValueError("training deal count does not match the recorded stratified sample count")
        independent_validation = provenance.get("independent_stratified_validation")
        if not isinstance(independent_validation, Mapping):
            raise ValueError("every training example must have independent validation provenance")
        if independent_validation.get("method") != "stratified_crossfit":
            raise ValueError("training labels need independent stratified cross-fit validation")
        validation_samples = independent_validation.get("samples_per_seat_hand_class")
        validation_seed = independent_validation.get("seed")
        validation_deals = independent_validation.get("total_physical_deals")
        if (
            isinstance(validation_samples, bool)
            or not isinstance(validation_samples, int)
            or validation_samples < 2
            or isinstance(validation_seed, bool)
            or not isinstance(validation_seed, int)
            or validation_seed < 0
            or isinstance(validation_deals, bool)
            or not isinstance(validation_deals, int)
            or validation_deals != 3 * len(HAND_CLASSES) * validation_samples
        ):
            raise ValueError("independent validation provenance is incomplete or inconsistent")
        validation_gap = float(independent_validation["mean_strategy_plugin_nash_conv"])
        validation_crossfit = float(independent_validation["mean_strategy_crossfit_br_gain"])
        if not math.isfinite(validation_gap) or validation_gap < 0.0 or not math.isfinite(validation_crossfit):
            raise ValueError("independent validation metrics must be finite and NashConv nonnegative")
        current_provenance = {
            "solver": solver,
            "sampling_method": provenance["sampling_method"],
            "solver_seeds": solver_seeds,
            "samples_per_hand_class_per_seed": sample_count,
            "independent_validation": {
                "method": independent_validation["method"],
                "samples_per_seat_hand_class": validation_samples,
                "seed": validation_seed,
            },
        }
        if shared_provenance is None:
            shared_provenance = current_provenance
        elif current_provenance != shared_provenance:
            raise ValueError("training labels mix solver seeds or sampling/validation configurations")

        hand_class = parse_hand_class(str(features["hand_class"]))
        hand_index = HAND_CLASSES.index(hand_class)
        example_key = (hand_index, stack_vector)  # type: ignore[arg-type]
        if example_key in seen_examples:
            raise ValueError(f"duplicate label for {hand_class} at BTN/SB/BB={stack_vector}")
        seen_examples.add(example_key)
        stack_vectors.add(stack_vector)  # type: ignore[arg-type]
        for action in ACTION_NAMES:
            target = row["targets"][action]
            frequency = float(target["mean_frequency"])
            label_sd = float(target["seed_standard_deviation"])
            if not math.isfinite(frequency) or not 0.0 <= frequency <= 1.0:
                raise ValueError("action frequencies must be finite values in [0, 1]")
            if not math.isfinite(label_sd) or label_sd < 0.0:
                raise ValueError("label seed deviations must be finite and nonnegative")
            grouped[action][hand_index].append((stack_vector, frequency, label_sd))  # type: ignore[arg-type]

    ordered_vectors = tuple(sorted(stack_vectors))
    if len(ordered_vectors) < 5 or not any(len(set(vector)) > 1 for vector in ordered_vectors):
        raise ValueError("training requires at least five vectors and at least one unequal-stack vector")
    stack_min = tuple(min(vector[seat] for vector in ordered_vectors) for seat in range(3))
    stack_max = tuple(max(vector[seat] for vector in ordered_vectors) for seat in range(3))
    if any(maximum <= minimum for minimum, maximum in zip(stack_min, stack_max)):
        raise ValueError("each seat must have at least two training stack depths")
    center = tuple((minimum + maximum) / 2.0 for minimum, maximum in zip(stack_min, stack_max))
    scale = tuple((maximum - minimum) / 2.0 for minimum, maximum in zip(stack_min, stack_max))

    def normalized_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple((value - center[index]) / scale[index] for index, value in enumerate(vector))  # type: ignore[return-value]

    errors: list[tuple[float, int]] = []
    baseline_errors: list[tuple[float, int]] = []
    total_label_sd: list[tuple[float, int]] = []
    errors_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    baseline_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    errors_by_vector: dict[tuple[float, float, float], list[tuple[float, int]]] = {
        vector: [] for vector in ordered_vectors
    }
    coefficients: list[tuple[tuple[float, float, float, float], ...]] = []
    for action in ACTION_NAMES:
        action_coefficients = []
        for hand_index, hand_class in enumerate(HAND_CLASSES):
            points = grouped[action][hand_index]
            by_vector = {vector: (frequency, label_sd) for vector, frequency, label_sd in points}
            if set(by_vector) != set(ordered_vectors):
                raise ValueError(f"hand class {hand_class} is missing a training stack vector for {action}")
            normalized_points = [
                (normalized_vector(vector), frequency, label_sd)
                for vector, frequency, label_sd in points
            ]
            action_coefficients.append(
                _fit_weighted_stack_surface(normalized_points, interaction_terms=interaction_terms)
            )
            combo_weight = combo_count(hand_class)
            for held_vector in ordered_vectors:
                fit_points = [point for point in normalized_points if point[0] != normalized_vector(held_vector)]
                intercept, *slopes = _fit_weighted_stack_surface(
                    fit_points,
                    interaction_terms=interaction_terms,
                )
                x = normalized_vector(held_vector)
                if interaction_terms:
                    btn, sb, bb = x
                    x = (*x, btn * sb, btn * bb, sb * bb)  # type: ignore[assignment]
                predicted = min(1.0, max(0.0, intercept + sum(slope * value for slope, value in zip(slopes, x))))
                frequency, label_sd = by_vector[held_vector]
                errors.append((abs(predicted - frequency), combo_weight))
                errors_by_action[action].append((abs(predicted - frequency), combo_weight))
                errors_by_vector[held_vector].append((abs(predicted - frequency), combo_weight))
                baseline_points = [value for vector, value, _ in points if vector != held_vector]
                baseline = sum(baseline_points) / len(baseline_points)
                baseline_error = abs(baseline - frequency)
                baseline_errors.append((baseline_error, combo_weight))
                baseline_by_action[action].append((baseline_error, combo_weight))
                total_label_sd.append((label_sd, combo_weight))
        coefficients.append(tuple(action_coefficients))

    error_weight = sum(weight for _, weight in errors)
    baseline_weight = sum(weight for _, weight in baseline_errors)
    validation = {
        "leave_one_stack_vector_out_combo_weighted_mae": sum(error * weight for error, weight in errors) / error_weight,
        "leave_one_stack_vector_out_combo_weighted_rmse": math.sqrt(
            sum(error * error * weight for error, weight in errors) / error_weight
        ),
        "constant_per_hand_baseline_combo_weighted_mae": sum(
            error * weight for error, weight in baseline_errors
        ) / baseline_weight,
        "mae_reduction_vs_constant_baseline": (
            sum(error * weight for error, weight in baseline_errors) / baseline_weight
            - sum(error * weight for error, weight in errors) / error_weight
        ),
        "mean_seed_label_standard_deviation": sum(
            value * weight for value, weight in total_label_sd
        ) / sum(weight for _, weight in total_label_sd),
    }
    for action in ACTION_NAMES:
        action_weight = sum(weight for _, weight in errors_by_action[action])
        validation[f"loo_mae_{action}"] = sum(
            error * weight for error, weight in errors_by_action[action]
        ) / action_weight
        validation[f"baseline_mae_{action}"] = sum(
            error * weight for error, weight in baseline_by_action[action]
        ) / action_weight
    for vector, vector_errors in errors_by_vector.items():
        vector_weight = sum(weight for _, weight in vector_errors)
        vector_key = "_".join(f"{value:g}".replace(".", "p") for value in vector)
        validation[f"loo_mae_at_btn_sb_bb_{vector_key}"] = sum(
            error * weight for error, weight in vector_errors
        ) / vector_weight

    return UnequalStackThreeMaxFrequencyModel(
        payouts=tuple(float(value) for value in payouts),
        stack_min_by_seat_bb=stack_min,  # type: ignore[arg-type]
        stack_max_by_seat_bb=stack_max,  # type: ignore[arg-type]
        stack_center_by_seat_bb=center,  # type: ignore[arg-type]
        stack_scale_by_seat_bb=scale,  # type: ignore[arg-type]
        coefficients=tuple(coefficients),
        training_stack_vectors_bb=ordered_vectors,
        training_examples=len(examples),
        validation=validation,
        training_provenance=shared_provenance or {},
        interaction_terms=interaction_terms,
    )


__all__ = [
    "ACTION_NAMES",
    "ThreeMaxFrequencyModel",
    "UnequalStackThreeMaxFrequencyModel",
    "train_three_max_frequency_model",
    "train_unequal_stack_three_max_frequency_model",
]
