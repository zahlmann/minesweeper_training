import json
import asyncio
import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from functools import partial

import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, TrajectoryGroup, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import ToolSpec
            

Cell = tuple[int, int]

PLAYING = "playing"
WON = "won"
LOST = "lost"
TOOL_NAME = "play_minesweeper"

SYSTEM_PROMPT = """Play Minesweeper by calling play_minesweeper.

Use one command per tool call:
- reveal ROW COL

Coordinates are zero-based. The first reveal is safe.
Reveal all safe cells without revealing a mine. The board is returned after every command.
The mines are placed so the board can be solved by logic without guessing.
Bad command syntax or unrelated tools end the rollout with minimum reward.
"""

@dataclass(frozen=True)
class GameConfig:
    rows: int
    cols: int
    mines: int
    seed: int

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be at least 1")
        if self.cols < 1:
            raise ValueError("cols must be at least 1")
        if self.mines < 1:
            raise ValueError("mines must be at least 1")
        if self.mines > self.max_mines:
            message = (
                f"mines must be at most {self.max_mines} "
                f"for a {self.rows}x{self.cols} board"
            )
            raise ValueError(message)

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def safe_cells(self) -> int:
        return self.cells - self.mines

    @property
    def first_reveal_safe_cells(self) -> int:
        return min(self.rows, 3) * min(self.cols, 3)

    @property
    def max_mines(self) -> int:
        return self.cells - self.first_reveal_safe_cells


@dataclass
class Game:
    config: GameConfig
    mines: set[Cell] = field(default_factory=set)
    revealed: set[Cell] = field(default_factory=set)
    status: str = PLAYING
    commands: int = 0
    invalid_commands: int = 0
    last_message: str = "new game initialized"

    @classmethod
    def new(cls, config: GameConfig) -> "Game":
        return cls(config=config)

    @property
    def total_safe_cells(self) -> int:
        return self.config.safe_cells

    @property
    def revealed_safe_cells(self) -> int:
        return len(self.revealed - self.mines)

    def reveal(self, cell: Cell) -> str:
        self.commands += 1

        if not self.in_bounds(cell):
            return self.invalid(f"{cell} is outside the board")
        if cell in self.revealed:
            return self.invalid(f"{cell} is already revealed")
        if not self.mines:
            self.place_mines(first_cell=cell)
        if cell in self.mines:
            return self.lose(cell)

        opened = self.open_area(cell)
        if self.revealed_safe_cells == self.total_safe_cells:
            self.status = WON
            self.last_message = f"won: revealed the final {opened} safe cell(s)"
        else:
            self.last_message = f"revealed {opened} safe cell(s) from {cell}"
        return self.render()

    def render(self) -> str:
        lines = [
            f"status: {self.status}",
            (
                f"rows: {self.config.rows} cols: {self.config.cols} mines: {self.config.mines} "
                f"revealed_safe: {self.revealed_safe_cells}/{self.total_safe_cells}"
            ),
            "coords: zero-based row col",
            "legend: # hidden, . clear, 1-8 adjacent mines",
            f"last: {self.last_message}",
            "",
        ]
        row_width = len(str(self.config.rows - 1))
        cell_width = len(str(self.config.cols - 1))
        row_label_width = len(f"row {self.config.rows - 1}:")
        header_gap = " " * (row_label_width - len("cols:") + 1)
        header_cells = " ".join(
            f"{col:>{cell_width}}" for col in range(self.config.cols)
        )
        lines.append(f"cols:{header_gap}{header_cells}")

        for row in range(self.config.rows):
            cells = [
                f"{self.visible((row, col)):>{cell_width}}"
                for col in range(self.config.cols)
            ]
            lines.append(f"row {row:>{row_width}}: " + " ".join(cells))

        return "\n".join(lines)

    def visible(self, cell: Cell) -> str:
        if cell in self.revealed:
            count = self.adjacent_mines(cell)
            return "." if count == 0 else str(count)
        return "#"

    def place_mines(self, first_cell: Cell) -> None:
        for attempt in range(1000):
            mines = self.sample_mines(first_cell, attempt)
            if self.can_solve_without_guessing(first_cell, mines):
                self.mines = mines
                return

        raise ValueError("could not generate a no-guess board")

    def sample_mines(self, first_cell: Cell, attempt: int) -> set[Cell]:
        all_cells = [
            (row, col)
            for row in range(self.config.rows)
            for col in range(self.config.cols)
        ]
        blocked = {first_cell, *self.neighbors(first_cell)}
        candidates = [cell for cell in all_cells if cell not in blocked]

        seed = f"{self.config.seed}:{first_cell[0]}:{first_cell[1]}:{attempt}"
        rng = random.Random(seed)
        return set(rng.sample(candidates, self.config.mines))

    def can_solve_without_guessing(self, first_cell: Cell, mines: set[Cell]) -> bool:
        revealed: set[Cell] = set()
        known_mines: set[Cell] = set()
        self.open_area_for_solver(first_cell, mines, revealed)

        while len(revealed) < self.total_safe_cells:
            made_progress = False

            for cell in list(revealed):
                hidden_neighbors = [
                    neighbor
                    for neighbor in self.neighbors(cell)
                    if neighbor not in revealed and neighbor not in known_mines
                ]
                if not hidden_neighbors:
                    continue

                known_neighbor_mines = len(
                    [
                        neighbor
                        for neighbor in self.neighbors(cell)
                        if neighbor in known_mines
                    ]
                )
                remaining_mines = (
                    self.adjacent_mines(cell, mines)
                    - known_neighbor_mines
                )

                if remaining_mines == 0:
                    for safe_cell in hidden_neighbors:
                        self.open_area_for_solver(safe_cell, mines, revealed)
                    made_progress = True
                    continue

                if remaining_mines == len(hidden_neighbors):
                    known_mines.update(hidden_neighbors)
                    made_progress = True

            if not made_progress:
                return False

        return True

    def open_area_for_solver(
        self,
        start: Cell,
        mines: set[Cell],
        revealed: set[Cell],
    ) -> None:
        queue: deque[Cell] = deque([start])

        while queue:
            cell = queue.popleft()
            if cell in revealed or cell in mines:
                continue

            revealed.add(cell)
            if self.adjacent_mines(cell, mines) == 0:
                queue.extend(
                    neighbor
                    for neighbor in self.neighbors(cell)
                    if neighbor not in revealed
                )

    def open_area(self, start: Cell) -> int:
        before = len(self.revealed)
        queue: deque[Cell] = deque([start])

        while queue:
            cell = queue.popleft()
            if cell in self.revealed or cell in self.mines:
                continue

            self.revealed.add(cell)
            if self.adjacent_mines(cell) == 0:
                queue.extend(
                    neighbor
                    for neighbor in self.neighbors(cell)
                    if neighbor not in self.revealed
                )

        return len(self.revealed) - before

    def lose(self, cell: Cell) -> str:
        self.revealed.add(cell)
        self.status = LOST
        self.last_message = f"BOOM: revealed a mine at {cell}"
        return self.render()

    def invalid(self, message: str) -> str:
        self.invalid_commands += 1
        self.last_message = f"INVALID: {message}"
        return self.render()

    def adjacent_mines(self, cell: Cell, mines: set[Cell] | None = None) -> int:
        mine_cells = self.mines if mines is None else mines
        return len(
            [
                neighbor
                for neighbor in self.neighbors(cell)
                if neighbor in mine_cells
            ]
        )

    def neighbors(self, cell: Cell) -> list[Cell]:
        row, col = cell
        cells = []
        for next_row in range(row - 1, row + 2):
            for next_col in range(col - 1, col + 2):
                neighbor = (next_row, next_col)
                if neighbor != cell and self.in_bounds(neighbor):
                    cells.append(neighbor)
        return cells

    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.config.rows and 0 <= col < self.config.cols


class MinesweeperEnv(Env):
    def __init__(self, renderer, rows=5, cols=5, mines=8):
        self.renderer = renderer
        self.config = GameConfig(rows=rows, cols=cols, mines=mines, seed=420)
        self.state = Game(self.config)

    async def get_state(self):
        return self.state.render()

    async def initial_observation(self):
        tools = self.renderer.create_conversation_prefix_with_tools([
            ToolSpec(
                name="reveal",
                description="Reveal a cell in the Minesweeper grid.",
                parameters={
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer"},
                        "col": {"type": "integer"},
                    },
                    "required": ["row", "col"],
                },
            )
        ])
        self.messages = [
            *tools,
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.state.render()}
        ]
        model_input = self.renderer.build_generation_prompt(self.messages)
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None):
        fallback_negative_reward = StepResult(
                reward=-1.0,
                episode_done=True,
                next_observation=tinker.ModelInput.from_ints([]),
                next_stop_condition=[],
            )

        message, termination = self.renderer.parse_response(action)
        print(message)
        if termination == "malformed":
            return fallback_negative_reward        

        self.messages.append(message)
        
        if not message.get("tool_calls"):
            return fallback_negative_reward

        for tool_call in message["tool_calls"]:
            if tool_call.function.name != "reveal":
                return fallback_negative_reward

            args = json.loads(tool_call.function.arguments)

            if "row" not in args or "col" not in args:
                return fallback_negative_reward
            
            row = args["row"]
            col = args["col"]

            try:
                cell = (int(row), int(col))
            except ValueError:
                return fallback_negative_reward

            self.state.reveal(cell)

        if self.state.status != PLAYING:
            reward = 1.0 if self.state.status == "WON" else -1.0
            done = True
        else:
            reward = 0.0
            done = False

        self.messages.append({
            "role": "tool",
            "name": tool_call.function.name,
            "content": self.state.render(),
            "tool_call_id": tool_call.function.id
        })
        next_obs = self.renderer.build_generation_prompt(self.messages)

        return StepResult(
            reward=reward,
            episode_done=done,
            next_observation=next_obs,
            next_stop_condition=self.renderer.get_stop_sequences(),
        )
    
async def main():
    MODEL_NAME = "openai/gpt-oss-20b"
    RENDERER_NAME = "gpt_oss_high_reasoning"
    GROUP_SIZE = 16
    LORA_RANK = 32
    MAX_TOKENS = 20000

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)


    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    policy = TinkerTokenCompleter(sampling_client, max_tokens=MAX_TOKENS, temperature=1.0)
    group_builder = ProblemGroupBuilder(
        env_thunk=partial(
            MinesweeperEnv,
            renderer,
        ),
        num_envs=GROUP_SIZE,
    )
    traj_group: TrajectoryGroup = await do_group_rollout(group_builder, policy)
    rewards = traj_group.get_total_rewards()
    print(f"Rewards per trajectory: {rewards}")
    print(f"Number of trajectories: {len(traj_group.trajectories_G)}")
    for i, (traj, reward) in enumerate(zip(traj_group.trajectories_G, rewards)):
        n_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
        print(f"  Trajectory {i}: reward={reward:.1f}, response_tokens={n_tokens}")

if __name__ == "__main__":
    asyncio.run(main())