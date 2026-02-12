import json
import random
import re
from pathlib import Path

from src.dataset.jwtd import prepare_jwtd
from src.dataset.model import CharCount

DATA_DIR = Path("data/char_count")
OUTPUT_FILE = DATA_DIR / "test.jsonl"
JWTD_FILE = Path("data/jwtd/test.jsonl")
ID_PREFIX = "char_count_wiki"


def generate_char_count_dataset(
    n_samples: int,
    target_length: int,
    length_variance: float,
    target_chars: list[str],
):
    samples: list[CharCount] = []
    prepare_jwtd()

    with open(JWTD_FILE, "r", encoding="utf-8") as f:
        jwtd_lines = f.readlines()

    random.shuffle(jwtd_lines)

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
                character = random.choice(target_chars)
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
        character = random.choice(target_chars)
        samples.append(
            {
                "id": f"{ID_PREFIX}_{count}",
                "text": current_block,
                "character": character,
                "count": current_block.count(character),
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_char_count():
    if not OUTPUT_FILE.exists():
        print("Generating CharCount dataset from JWTD...")
        generate_char_count_dataset(
            500,
            150,
            0.2,
            [
                "が",
                "は",
                "を",
                "に",
                "の",
                "も",
                "た",
                "て",
                "だ",
                "る",
                "。",
                "、",
                "日",
                "本",
                "学",
                "者",
            ],
        )
        print(f"CharCount dataset generated at {OUTPUT_FILE}")
