import json
import random
import re
from pathlib import Path

from src.dataset.jwtd import prepare_jwtd
from src.dataset.model import CharCount

DATA_DIR = Path("data/char_count")
JWTD_FILE = Path("data/jwtd/test.jsonl")
ID_PREFIX = "char_count_wiki"
DEFAULT_TARGET_LENGTH = 150
DEFAULT_LENGTH_VARIANCE = 0.2
DEFAULT_NUM_SAMPLES = 500
TARGET_CHARS = ["が", "は", "を", "に", "の", "も", "た", "て", "だ", "る", "。", "、", "日", "本", "学", "者"]


def get_char_count_output_file(length_multiplier: int):
    return DATA_DIR / f"test_m{length_multiplier}.jsonl"


def generate_char_count_dataset(n_samples: int, target_length: int, length_variance: float, target_chars: list[str], output_file: Path, seed: int):
    samples: list[CharCount] = []
    prepare_jwtd()
    rng = random.Random(seed)

    with open(JWTD_FILE, "r", encoding="utf-8") as f:
        jwtd_lines = f.readlines()

    rng.shuffle(jwtd_lines)

    min_len = int(target_length * (1 - length_variance))
    max_len = int(target_length * (1 + length_variance))

    count = 0
    current_block = ""
    for line in jwtd_lines:
        if count >= n_samples:
            break

        data = json.loads(line)
        text = data["pre_text"]
        text = re.sub(r"\s+", " ", text).strip()

        if len(current_block) + len(text) <= max_len:
            current_block += text
        else:
            if len(current_block) >= min_len:
                character = rng.choice(target_chars)
                samples.append(
                    {
                        "id": f"{ID_PREFIX}_{count}",
                        "text": current_block,
                        "character": character,
                        "count": current_block.count(character),
                    }
                )
                count += 1
            current_block = text

    if count < n_samples and len(current_block) >= min_len:
        character = rng.choice(target_chars)
        samples.append(
            {
                "id": f"{ID_PREFIX}_{count}",
                "text": current_block,
                "character": character,
                "count": current_block.count(character),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_char_count(length_multiplier: int, seed: int):
    if length_multiplier < 1:
        raise ValueError("length_multiplier must be an integer >= 1.")

    output_file = get_char_count_output_file(length_multiplier)
    if not output_file.exists():
        print("Generating CharCount dataset from JWTD...")
        generate_char_count_dataset(
            n_samples=DEFAULT_NUM_SAMPLES,
            target_length=DEFAULT_TARGET_LENGTH * length_multiplier,
            length_variance=DEFAULT_LENGTH_VARIANCE,
            target_chars=TARGET_CHARS,
            output_file=output_file,
            seed=seed,
        )
        print(f"CharCount dataset generated at {output_file}")
