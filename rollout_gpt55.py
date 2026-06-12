import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from rollout import Game, GameConfig, LOST, PLAYING, WON


MODEL = "gpt-5.5"
REASONING_EFFORT = "medium"
RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_OUTPUT_PATH = "artifacts/gpt55_rollouts.json"

SYSTEM_PROMPT = """Play Minesweeper by calling the reveal tool exactly once per turn.

Coordinates are zero-based. The first reveal has already been performed safely.
Reveal all safe cells without revealing a mine.
The board is guaranteed to be solvable by logic without guessing.
Only the reveal tool call controls the game.
Any text you write is logged but ignored.
"""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, usage: dict[str, Any]) -> None:
        if "cached_input_tokens" in usage or "reasoning_tokens" in usage:
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.cached_input_tokens += int(usage.get("cached_input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
            self.reasoning_tokens += int(usage.get("reasoning_tokens", 0))
            return

        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}

        self.input_tokens += int(usage.get("input_tokens", 0))
        self.cached_input_tokens += int(input_details.get("cached_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.reasoning_tokens += int(output_details.get("reasoning_tokens", 0))


@dataclass(frozen=True)
class ModelTurn:
    row: int
    col: int
    tool_call_id: str | None
    text_output: str
    usage: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-5.5 medium on one Minesweeper configuration."
    )
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--mines", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_chatgpt_auth() -> tuple[str, str | None]:
    auth_path = Path(
        os.environ.get("CODEX_AUTH_FILE", "~/.codex/auth.json")
    ).expanduser()

    if not auth_path.exists():
        raise FileNotFoundError(
            f"Codex auth file not found: {auth_path}. Run `codex login` first."
        )

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    tokens = auth.get("tokens") or {}
    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")

    if not access_token:
        raise ValueError(f"Codex auth file has no access token: {auth_path}")

    return access_token, account_id


def build_reveal_tool(rows: int, cols: int) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "reveal",
        "description": "Reveal one hidden cell in the Minesweeper grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": rows - 1,
                },
                "col": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": cols - 1,
                },
            },
            "required": ["row", "col"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def parse_sse_events(stream_text: str) -> list[dict[str, Any]]:
    events = []
    data_lines: list[str] = []

    def append_event() -> None:
        if not data_lines:
            return

        data = "\n".join(data_lines)
        data_lines.clear()
        if data == "[DONE]":
            return

        events.append(json.loads(data))

    for line in stream_text.splitlines():
        if not line:
            append_event()
            continue

        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip())

    append_event()
    return events


def text_from_output_items(output_items: list[dict[str, Any]]) -> str:
    text_parts = []

    for item in output_items:
        if item.get("type") != "message":
            continue

        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                text_parts.append(content.get("text", ""))

    return "\n".join(part for part in text_parts if part)


def parse_response_stream(stream_text: str) -> ModelTurn:
    completed_output: list[dict[str, Any]] | None = None
    completed_usage: dict[str, Any] = {}
    streamed_output_items = []
    streamed_text_parts = []

    for event in parse_sse_events(stream_text):
        event_type = event.get("type")

        if event_type == "response.output_text.done":
            streamed_text_parts.append(event.get("text", ""))
            continue

        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                streamed_output_items.append(item)
            continue

        if event_type == "response.completed":
            response = event.get("response") or {}
            completed_output = response.get("output") or []
            completed_usage = response.get("usage") or {}
            continue

        if event_type == "response.failed":
            response = event.get("response") or {}
            error = response.get("error") or event
            raise RuntimeError(f"GPT-5.5 response failed: {error}")

        if event_type == "error":
            raise RuntimeError(f"GPT-5.5 stream failed: {event}")

    output_items = completed_output or streamed_output_items
    function_calls = [
        item for item in output_items if item.get("type") == "function_call"
    ]

    if len(function_calls) != 1:
        raise ValueError(
            "GPT-5.5 must return exactly one function call; "
            f"received {len(function_calls)}"
        )

    function_call = function_calls[0]
    if function_call.get("name") != "reveal":
        raise ValueError(
            "GPT-5.5 called an unsupported tool: "
            f"{function_call.get('name')!r}"
        )

    arguments = function_call.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError("The reveal tool arguments must be a JSON object")
    if "row" not in arguments or "col" not in arguments:
        raise ValueError("The reveal tool call must include row and col")

    text_output = "\n".join(part for part in streamed_text_parts if part)
    if not text_output:
        text_output = text_from_output_items(output_items)

    return ModelTurn(
        row=int(arguments["row"]),
        col=int(arguments["col"]),
        tool_call_id=function_call.get("call_id"),
        text_output=text_output,
        usage=completed_usage,
    )


class ChatGPTCodexClient:
    def __init__(self, rows: int, cols: int) -> None:
        access_token, account_id = load_chatgpt_auth()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        self.reveal_tool = build_reveal_tool(rows, cols)
        self.responses_url = os.environ.get(
            "CHATGPT_RESPONSES_URL",
            RESPONSES_URL,
        )
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(300, connect=30),
        )

    async def choose_move(self, board: str) -> ModelTurn:
        payload = {
            "model": MODEL,
            "instructions": SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Call reveal once for the next move on this "
                        f"authoritative board.\n\n{board}"
                    ),
                }
            ],
            "reasoning": {"effort": REASONING_EFFORT},
            "tools": [self.reveal_tool],
            "tool_choice": {"type": "function", "name": "reveal"},
            "parallel_tool_calls": False,
            "stream": True,
            "store": False,
        }

        response = None
        for attempt in range(5):
            response = await self.client.post(self.responses_url, json=payload)

            if (
                response.status_code == 429
                and "usage_limit_reached" in response.text
            ):
                break

            should_retry = response.status_code == 429 or response.status_code >= 500
            if not should_retry or attempt == 4:
                break

            await asyncio.sleep(2**attempt)

        if response is None:
            raise RuntimeError("GPT-5.5 request did not produce a response")

        if response.is_error:
            body = response.text[:1000]
            raise RuntimeError(
                f"GPT-5.5 request failed with HTTP {response.status_code}: {body}"
            )

        return parse_response_stream(response.text)

    async def aclose(self) -> None:
        await self.client.aclose()


def center_cell(config: GameConfig) -> tuple[int, int]:
    return config.rows // 2, config.cols // 2


async def run_rollout(
    rollout_id: int,
    base_config: GameConfig,
    client: ChatGPTCodexClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    config = GameConfig(
        rows=base_config.rows,
        cols=base_config.cols,
        mines=base_config.mines,
        seed=rollout_id,
    )
    game = Game.new(config)
    opening_cell = center_cell(config)
    game.reveal(opening_cell)

    initial_board = game.render()
    usage = TokenUsage()
    moves = []
    started_at = time.monotonic()
    error = None

    while game.status == PLAYING:
        try:
            async with semaphore:
                model_turn = await client.choose_move(game.render())
        except Exception as request_error:
            error = f"{type(request_error).__name__}: {request_error}"
            break

        turn = len(moves) + 1
        usage.add(model_turn.usage)
        game.reveal((model_turn.row, model_turn.col))
        moves.append(
            {
                "turn": turn,
                "tool": "reveal",
                "tool_call_id": model_turn.tool_call_id,
                "arguments": {
                    "row": model_turn.row,
                    "col": model_turn.col,
                },
                "text_output": model_turn.text_output,
                "status": game.status,
                "last_message": game.last_message,
                "usage": model_turn.usage,
            }
        )

    if game.status == WON:
        reward = 1.0
        final_status = WON
    elif game.status == LOST:
        reward = -1.0
        final_status = LOST
    else:
        reward = -1.0
        final_status = "error"

    return {
        "rollout_id": rollout_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "config": asdict(config),
        "opening_cell": list(opening_cell),
        "initial_board": initial_board,
        "final_board": game.render(),
        "final_status": final_status,
        "reward": reward,
        "turns": len(moves),
        "moves": moves,
        "invalid_moves": game.invalid_commands,
        "error": error,
        "usage": asdict(usage),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    usage = TokenUsage()
    for result in results:
        usage.add(result["usage"])

    wins = sum(result["final_status"] == WON for result in results)
    losses = sum(result["final_status"] == LOST for result in results)
    errors = sum(result["final_status"] == "error" for result in results)
    completed = wins + losses
    total_turns = sum(result["turns"] for result in results)
    total_elapsed = sum(result["elapsed_seconds"] for result in results)

    return {
        "rollouts": len(results),
        "wins": wins,
        "losses": losses,
        "errors": errors,
        "completed_rollouts": completed,
        "win_rate_all_attempts": wins / len(results) if results else 0.0,
        "win_rate_completed": wins / completed if completed else 0.0,
        "mean_reward": (
            sum(result["reward"] for result in results) / len(results)
            if results
            else 0.0
        ),
        "mean_turns": total_turns / len(results) if results else 0.0,
        "mean_elapsed_seconds": total_elapsed / len(results) if results else 0.0,
        "usage": asdict(usage),
    }


async def main() -> None:
    args = parse_args()

    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    base_config = GameConfig(
        rows=args.rows,
        cols=args.cols,
        mines=args.mines,
        seed=0,
    )
    client = ChatGPTCodexClient(rows=args.rows, cols=args.cols)
    semaphore = asyncio.Semaphore(args.concurrency)

    try:
        tasks = [
            asyncio.create_task(
                run_rollout(
                    rollout_id=rollout_id,
                    base_config=base_config,
                    client=client,
                    semaphore=semaphore,
                )
            )
            for rollout_id in range(args.num_rollouts)
        ]

        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            print(
                f"rollout={result['rollout_id']:02d} "
                f"board={result['config']['rows']}x{result['config']['cols']} "
                f"mines={result['config']['mines']} "
                f"status={result['final_status']} "
                f"turns={result['turns']} "
                f"tokens={result['usage']['input_tokens']}+"
                f"{result['usage']['output_tokens']}"
            )
    finally:
        await client.aclose()

    results.sort(key=lambda result: result["rollout_id"])
    summary = summarize(results)
    output = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "context_mode": "current_board_only",
        "tool": build_reveal_tool(args.rows, args.cols),
        "config": {
            "rows": args.rows,
            "cols": args.cols,
            "mines": args.mines,
        },
        "summary": summary,
        "rollouts": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
