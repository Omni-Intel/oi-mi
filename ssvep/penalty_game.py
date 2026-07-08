"""State model for the SSVEP football penalty shootout game."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Team:
    """Display metadata for a selectable football team."""

    name: str
    flag: str


@dataclass(frozen=True, slots=True)
class ShotDirection:
    """One shootable goal area."""

    key: str
    label: str
    frequency_hz: float


@dataclass(frozen=True, slots=True)
class ShotResult:
    """Outcome of one user penalty."""

    direction: ShotDirection
    keeper_direction: ShotDirection
    goal: bool
    opponent_goal: bool
    user_score: int
    opponent_score: int
    round_index: int
    complete: bool


TEAMS: tuple[Team, ...] = (
    Team("China", "🇨🇳"),
    Team("Japan", "🇯🇵"),
    Team("Brazil", "🇧🇷"),
    Team("Argentina", "🇦🇷"),
    Team("France", "🇫🇷"),
    Team("Germany", "🇩🇪"),
    Team("England", "🏴"),
    Team("Spain", "🇪🇸"),
    Team("USA", "🇺🇸"),
    Team("Portugal", "🇵🇹"),
    Team("Korea", "🇰🇷"),
    Team("Italy", "🇮🇹"),
)

DIRECTIONS: tuple[ShotDirection, ...] = (
    ShotDirection("top_left", "左上", 10.0),
    ShotDirection("top_right", "右上", 12.0),
    ShotDirection("bottom_left", "左下", 15.0),
    ShotDirection("bottom_right", "右下", 18.0),
)


class PenaltyGame:
    """Five-round penalty shootout state machine."""

    def __init__(
        self,
        *,
        user_team: Team,
        opponent_team: Team,
        rounds: int = 5,
        keeper_save_probability: float = 0.30,
        opponent_goal_probability: float = 0.70,
        seed: int | None = None,
    ) -> None:
        if rounds <= 0:
            raise ValueError("rounds must be positive.")
        self.user_team = user_team
        self.opponent_team = opponent_team
        self.rounds = int(rounds)
        self.keeper_save_probability = float(keeper_save_probability)
        self.opponent_goal_probability = float(opponent_goal_probability)
        self._rng = random.Random(seed)
        self.user_score = 0
        self.opponent_score = 0
        self.round_index = 0
        self.history: list[ShotResult] = []

    @property
    def complete(self) -> bool:
        return self.round_index >= self.rounds

    def shoot(self, direction_key: str) -> ShotResult:
        """Resolve one user shot and one simulated opponent shot."""

        if self.complete:
            raise RuntimeError("Penalty shootout is already complete.")
        direction = direction_by_key(direction_key)
        keeper_direction = self._choose_keeper_direction(direction)
        goal = keeper_direction.key != direction.key
        if goal:
            self.user_score += 1

        opponent_goal = self._rng.random() < self.opponent_goal_probability
        if opponent_goal:
            self.opponent_score += 1

        self.round_index += 1
        result = ShotResult(
            direction=direction,
            keeper_direction=keeper_direction,
            goal=goal,
            opponent_goal=opponent_goal,
            user_score=self.user_score,
            opponent_score=self.opponent_score,
            round_index=self.round_index,
            complete=self.complete,
        )
        self.history.append(result)
        return result

    def _choose_keeper_direction(self, user_direction: ShotDirection) -> ShotDirection:
        if self._rng.random() < self.keeper_save_probability:
            return user_direction
        other_directions = [direction for direction in DIRECTIONS if direction.key != user_direction.key]
        return self._rng.choice(other_directions)


def direction_by_key(key: str) -> ShotDirection:
    for direction in DIRECTIONS:
        if direction.key == key:
            return direction
    raise ValueError(f"Unknown shot direction: {key}")


def team_by_name(name: str) -> Team:
    for team in TEAMS:
        if team.name == name:
            return team
    raise ValueError(f"Unknown team: {name}")
