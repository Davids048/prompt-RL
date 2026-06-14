"""Convert a plain PickScore prompt text file into Slime JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_prompts(source_path: Path, output_path: Path, *, prompt_id_prefix: str) -> int:
    """Write Slime-compatible records while preserving source-line metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            prompt = line.rstrip("\n")
            if not prompt.strip():
                continue
            count += 1
            prompt_id = f"{prompt_id_prefix}_{count:06d}"
            record = {
                "prompt_id": prompt_id,
                "source_path": str(source_path),
                "source_line": line_number,
                "original_prompt": prompt,
                "metadata": {
                    "prompt_id": prompt_id,
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "original_prompt": prompt,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Plain text prompt file, one prompt per line.")
    parser.add_argument("--output", required=True, type=Path, help="Destination Slime JSONL path.")
    parser.add_argument(
        "--prompt-id-prefix",
        default="diffusion_nft_pickscore_train",
        help="Prefix for generated prompt_id values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_prompts(args.source, args.output, prompt_id_prefix=args.prompt_id_prefix)
    print(f"wrote {count} prompts to {args.output}")


if __name__ == "__main__":
    main()
