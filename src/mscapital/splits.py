"""Explicit inclusive month splits used by every protocol-v2 experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonthRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("month range start must be <= end")

    def contains(self, months):
        return (months >= self.start) & (months <= self.end)

    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    train: MonthRange
    valid: MonthRange

    def validate(self) -> None:
        if self.train.end >= self.valid.start:
            raise ValueError(f"{self.name}: train and valid months overlap")


@dataclass(frozen=True)
class NestedSplit:
    name: str
    inner_train: MonthRange
    inner_tune: MonthRange
    refit_train: MonthRange
    outer_valid: MonthRange

    def validate(self) -> None:
        TemporalSplit(self.name, self.refit_train, self.outer_valid).validate()
        TemporalSplit(f"{self.name}-inner", self.inner_train, self.inner_tune).validate()
        if self.inner_tune.end >= self.refit_train.end + 1:
            raise ValueError(f"{self.name}: inner tune must precede refit end")


def _m(start: int, end: int) -> MonthRange:
    return MonthRange(start, end)


OUTER_SPLITS = {
    "PSEUDO": TemporalSplit("PSEUDO", _m(0, 32), _m(33, 70)),
    "H2": TemporalSplit("H2", _m(0, 40), _m(51, 60)),
    "T3": TemporalSplit("T3", _m(0, 50), _m(51, 60)),
    "T4": TemporalSplit("T4", _m(0, 50), _m(61, 70)),
}

NESTED_SPLITS = {
    "PSEUDO": NestedSplit("PSEUDO", _m(0, 20), _m(21, 32), _m(0, 32), _m(33, 70)),
    "H2": NestedSplit("H2", _m(0, 30), _m(31, 40), _m(0, 40), _m(51, 60)),
    "T3": NestedSplit("T3", _m(0, 40), _m(41, 50), _m(0, 50), _m(51, 60)),
    "T4": NestedSplit("T4", _m(0, 40), _m(41, 50), _m(0, 50), _m(61, 70)),
}

ROLLING_WINDOWS = (
    ("m21_30", _m(0, 20), _m(21, 30)),
    ("m31_40", _m(0, 30), _m(31, 40)),
    ("m41_50", _m(0, 40), _m(41, 50)),
    ("m51_60", _m(0, 50), _m(51, 60)),
    ("m61_70", _m(0, 60), _m(61, 70)),
)

for _split in OUTER_SPLITS.values():
    _split.validate()
for _split in NESTED_SPLITS.values():
    _split.validate()


def visible_oof_end(outer_name: str) -> int:
    try:
        return {"PSEUDO": 32, "H2": 40, "T3": 50, "T4": 50}[outer_name]
    except KeyError as exc:
        raise KeyError(f"unknown outer split: {outer_name}") from exc
