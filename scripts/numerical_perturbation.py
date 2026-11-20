"""
scripts/numerical_perturbation.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import re
import json
import random
import shutil
import hashlib
from pathlib import Path
from scripts.numeric_patterns import detect_clinical_numbers

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with (PROJECT_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CLEAN_DIR = Path(CONFIG["corpora"]["clean"])
CORRUPTED_ROOT = Path(CONFIG["paths"]["corpus_root"]) / "corrupted"
LOG_DIR = Path(CONFIG["paths"]["output_root"]) / "corruption_logs"

SEED = 42

def get_article_seed(article_id: str) -> int:

    seed_string = f"{SEED}:{article_id}"

    digest = hashlib.sha256(
        seed_string.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16
    )
CORRUPTION_RATES = [
    0.00,
    0.10,
    0.25,
    0.50,
    1.00,
]


def plausible_round(
    value: float,
    original_text: str,
    pattern_type: str,
) -> str:

    if "." in original_text:

        decimal_places = len(
            original_text.split(".")[1]
        )

        rounded = round(
            value,
            decimal_places
        )

    else:
        decimal_places = 0

        rounded = round(value)


    if pattern_type in {
        "age",
        "hyphenated_age",
    }:

        rounded = min(
            100,
            max(1, rounded)
        )


    else:

        rounded = max(
            0.1,
            rounded
        )


    if decimal_places > 0:

        return (
            f"{rounded:.{decimal_places}f}"
        )

    return str(
        int(round(rounded))
    )


def perturb_count_text(
    original_text: str,
    factor: float,
) -> str:

    plain = original_text.replace(",", "")
    value = float(plain)
    perturbed = max(1.0, value * factor)

    if "." in plain:
        decimal_places = len(plain.split(".")[1])
        rounded = round(perturbed, decimal_places)
        result = f"{rounded:.{decimal_places}f}"
    else:
        rounded = int(round(perturbed))
        result = str(rounded)

    if "," in original_text:
        if "." in result:
            integer_part, decimal_part = result.split(".", 1)
            result = f"{int(integer_part):,}.{decimal_part}"
        else:
            result = f"{int(result):,}"

    return result

def perturb_text(
    text: str,
    article_id: str,
):


    candidates = (
        detect_clinical_numbers(text)
    )

    audit_entries = []
    replacements = []


    for candidate in candidates:

        pattern_type = (
            candidate["pattern_type"]
        )

        groups = (
            candidate["groups"]
        )

        original_full = (
            candidate["original_text"]
        )

        start = (
            candidate["start"]
        )

        end = (
            candidate["end"]
        )


        if pattern_type == "epidemiology_numeric_ratio":

            numerator_text = groups["numerator"]
            denominator_text = groups["denominator"]
            ratio_separator = groups["ratio_separator"]

            max_attempts = 20

            for _ in range(max_attempts):

                factor = random.uniform(
                    0.5,
                    2.0
                )

                new_denominator = perturb_count_text(
                    denominator_text,
                    factor,
                )

                if new_denominator != denominator_text:
                    break

            else:
                original_denominator = int(
                    denominator_text.replace(",", "")
                )

                forced_denominator = (
                    original_denominator + 1
                )

                new_denominator = f"{forced_denominator:,}"

                factor = (
                    forced_denominator
                    / original_denominator
                    if original_denominator != 0
                    else None
                )

            new_full = (
                numerator_text
                + ratio_separator
                + new_denominator
            )

            replacements.append({
                "start": start,
                "end": end,
                "new_text": new_full,
            })

            audit_entries.append({
                "article_id": article_id,
                "pattern_type": pattern_type,
                "original": original_full,
                "perturbed": new_full,
                "original_numerator": float(numerator_text),
                "original_denominator": float(
                    denominator_text.replace(",", "")
                ),
                "perturbed_denominator": float(
                    new_denominator.replace(",", "")
                ),
                "factor": factor,
                "start_char": start,
                "end_char": end,
            })

            continue

        if pattern_type == "epidemiology_in_range":

            numerator_text = groups["numerator"]
            ratio_word = groups["ratio_word"]
            denominator1_text = groups["denominator1"]
            denominator2_text = groups["denominator2"]
            separator = groups["separator"]

            max_attempts = 20

            for _ in range(max_attempts):

                factor = random.uniform(
                    0.5,
                    2.0
                )

                new_denominator1 = perturb_count_text(
                    denominator1_text,
                    factor,
                )

                new_denominator2 = perturb_count_text(
                    denominator2_text,
                    factor,
                )

                if (
                    new_denominator1 != denominator1_text
                    or new_denominator2 != denominator2_text
                ):
                    break

            else:
                factor = 1.5

                new_denominator1 = perturb_count_text(
                    denominator1_text,
                    factor,
                )

                new_denominator2 = perturb_count_text(
                    denominator2_text,
                    factor,
                )

            new_full = (
                numerator_text
                + ratio_word
                + new_denominator1
                + separator
                + new_denominator2
            )

            replacements.append({
                "start": start,
                "end": end,
                "new_text": new_full,
            })

            audit_entries.append({
                "article_id": article_id,
                "pattern_type": pattern_type,
                "original": original_full,
                "perturbed": new_full,
                "original_numerator": float(numerator_text),
                "original_denominator_1": float(
                    denominator1_text.replace(",", "")
                ),
                "original_denominator_2": float(
                    denominator2_text.replace(",", "")
                ),
                "perturbed_denominator_1": float(
                    new_denominator1.replace(",", "")
                ),
                "perturbed_denominator_2": float(
                    new_denominator2.replace(",", "")
                ),
                "factor": factor,
                "start_char": start,
                "end_char": end,
            })

            continue


        if (
            groups.get("value1") is not None
            and groups.get("value2") is not None
        ):

            value1_text = (
                groups["value1"]
            )

            value2_text = (
                groups["value2"]
            )

            value1 = float(
                value1_text
            )

            value2 = float(
                value2_text
            )

            separator = (
                groups["separator"]
            )

            space = (
                groups.get("space") or ""
            )

            unit = (
                groups.get("unit") or ""
            )

            max_attempts = 20

            for _ in range(max_attempts):

                factor = random.uniform(
                    0.5,
                    2.0
                )

                perturbed1 = (
                    value1
                    * factor
                )

                perturbed2 = (
                    value2
                    * factor
                )

                new1 = plausible_round(
                    perturbed1,
                    value1_text,
                    pattern_type,
                )

                new2 = plausible_round(
                    perturbed2,
                    value2_text,
                    pattern_type,
                )

                percentage_range_valid = True

                if (
                    pattern_type == "percentage_range"
                    and 0 <= value1 <= 100
                    and 0 <= value2 <= 100
                ):
                    percentage_range_valid = (
                        0 <= float(new1) <= 100
                        and 0 <= float(new2) <= 100
                    )

                year_range_valid = True

                normalized_unit = unit.lower().strip()

                if (
                    normalized_unit
                    in {
                        "year",
                        "years",
                        "yr",
                        "yrs",
                    }
                    and 1 <= value1 <= 100
                    and 1 <= value2 <= 100
                ):
                    year_range_valid = (
                        1 <= float(new1) <= 100
                        and 1 <= float(new2) <= 100
                    )

                if (
                    (
                        float(new1) != value1
                        or float(new2) != value2
                    )
                    and percentage_range_valid
                    and year_range_valid
                ):
                    break

            else:
                fallback_factors = [
                    0.99,
                    1.01,
                    0.95,
                    1.05,
                    0.90,
                    1.10,
                    0.75,
                    1.25,
                    0.50,
                    1.50,
                    2.00,
                ]

                fallback_found = False

                for fallback_factor in fallback_factors:

                    fallback1 = plausible_round(
                        value1 * fallback_factor,
                        value1_text,
                        pattern_type,
                    )

                    fallback2 = plausible_round(
                        value2 * fallback_factor,
                        value2_text,
                        pattern_type,
                    )

                    percentage_range_valid = True

                    if (
                        pattern_type == "percentage_range"
                        and 0 <= value1 <= 100
                        and 0 <= value2 <= 100
                    ):
                        percentage_range_valid = (
                            0 <= float(fallback1) <= 100
                            and 0 <= float(fallback2) <= 100
                        )

                    year_range_valid = True

                    if (
                        normalized_unit
                        in {
                            "year",
                            "years",
                            "yr",
                            "yrs",
                        }
                        and 1 <= value1 <= 100
                        and 1 <= value2 <= 100
                    ):
                        year_range_valid = (
                            1 <= float(fallback1) <= 100
                            and 1 <= float(fallback2) <= 100
                        )

                    if (
                        (
                            float(fallback1) != value1
                            or float(fallback2) != value2
                        )
                        and percentage_range_valid
                        and year_range_valid
                    ):
                        factor = fallback_factor
                        new1 = fallback1
                        new2 = fallback2
                        fallback_found = True
                        break

                if not fallback_found:
                    raise RuntimeError(
                        "Unable to generate a valid multi-value "
                        f"perturbation for article '{article_id}': "
                        f"'{original_full}' "
                        f"(pattern={pattern_type})"
                    )

            prefix = (
                groups.get("prefix") or ""
            )

            new_full = (
                prefix
                + new1
                + separator
                + new2
                + space
                + unit
            )

            replacements.append({
                "start": start,
                "end": end,
                "new_text": new_full,
            })

            audit_entries.append({
                "article_id": article_id,
                "pattern_type": pattern_type,

                "original": original_full,
                "perturbed": new_full,

                "original_value_1": value1,
                "original_value_2": value2,

                "perturbed_value_1": float(new1),
                "perturbed_value_2": float(new2),

                "factor": factor,

                "start_char": start,
                "end_char": end,
            })

            continue


        value_text = (
            groups.get("value")
        )

        if value_text is None:
            continue

        original_value = float(
            value_text
        )

        space = (
            groups.get("space") or ""
        )

        unit = (
            groups.get("unit") or ""
        )


        max_attempts = 20

        if original_value == 0:

            if "." in value_text:

                decimal_places = len(
                    value_text.split(".")[1]
                )

                new_value = (
                    f"{10 ** (-decimal_places):.{decimal_places}f}"
                )

            else:

                new_value = "1"

            factor = None

        else:

            for _ in range(max_attempts):

                factor = random.uniform(
                    0.5,
                    2.0
                )

                perturbed_value = (
                    original_value
                    * factor
                )

                new_value = plausible_round(
                    perturbed_value,
                    value_text,
                    pattern_type,
                )

                percentage_valid = True

                if (
                    pattern_type == "percentage"
                    and 0 <= original_value <= 100
                ):
                    percentage_valid = (
                        0 <= float(new_value) <= 100
                    )

                if (
                    float(new_value) != original_value
                    and percentage_valid
                ):
                    break

            else:

                if (
                    pattern_type == "percentage"
                    and 0 <= original_value <= 100
                ):
                    if original_value >= 1:
                        forced_value = (
                            original_value - 1
                        )
                    else:
                        forced_value = min(
                            100,
                            original_value + 0.1
                        )

                elif original_value >= 2:

                    forced_value = (
                        original_value + 1
                    )

                else:

                    forced_value = (
                        original_value * 1.5
                    )

                new_value = plausible_round(
                    forced_value,
                    value_text,
                    pattern_type,
                )

                factor = (
                    float(new_value)
                    / original_value
                )

        prefix = (
            groups.get("prefix") or ""
        )

        suffix = (
            groups.get("suffix") or ""
        )

        new_full = (
            prefix
            + new_value
            + space
            + unit
            + suffix
        )

        replacements.append({
            "start": start,
            "end": end,
            "new_text": new_full,
        })

        audit_entries.append({
            "article_id": article_id,
            "pattern_type": pattern_type,

            "original": original_full,
            "perturbed": new_full,

            "original_value": original_value,
            "perturbed_value": float(
                new_value
            ),

            "factor": factor,

            "start_char": start,
            "end_char": end,
        })


    replacements.sort(
        key=lambda item: item["start"],
        reverse=True,
    )

    modified_text = text

    for replacement in replacements:

        start = (
            replacement["start"]
        )

        end = (
            replacement["end"]
        )

        new_text = (
            replacement["new_text"]
        )

        modified_text = (
            modified_text[:start]
            + new_text
            + modified_text[end:]
        )

    return (
        modified_text,
        audit_entries,
    )


def generate_variant(
    rate: float,
    selected_files: set,
):

    rate_label = int(rate * 100)

    output_dir = (
        CORRUPTED_ROOT
        / f"numerical_{rate_label:03d}"
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing numerical corpus: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    audit_log = []

    clean_files = sorted(
        CLEAN_DIR.glob("*.txt")
    )

    for source_path in clean_files:

        destination_path = (
            output_dir
            / source_path.name
        )

        text = source_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if source_path.name not in selected_files:

            destination_path.write_text(
                text,
                encoding="utf-8",
                newline="\r\n"
            )

            continue

        random.seed(
            get_article_seed(source_path.name)
        )

        modified_text, article_log = perturb_text(
            text,
            article_id=source_path.name
        )

        destination_path.write_text(
            modified_text,
            encoding="utf-8",
            newline="\r\n"
        )

        audit_log.extend(
            article_log
        )

    audit_path = (
        LOG_DIR
        / f"numerical_{rate_label:03d}_audit.json"
    )

    with audit_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit_log,
            f,
            ensure_ascii=False,
            indent=2
        )


def main():

    random.seed(SEED)

    clean_files = sorted(
        CLEAN_DIR.glob("*.txt")
    )

    filenames = [
        path.name
        for path in clean_files
    ]

    total_articles = len(filenames)


    shuffled_files = filenames.copy()

    random.shuffle(
        shuffled_files
    )

    for rate in CORRUPTION_RATES:

        n_selected = round(
            total_articles * rate
        )

        selected_files = set(
            shuffled_files[:n_selected]
        )

        generate_variant(
            rate,
            selected_files
        )


def main_numerical_perturbation(config_path):
    global CONFIG, CLEAN_DIR, CORRUPTED_ROOT, LOG_DIR
    with Path(config_path).open("r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    CLEAN_DIR = Path(CONFIG["corpora"]["clean"])
    CORRUPTED_ROOT = Path(CONFIG["paths"]["corpus_root"]) / "corrupted"
    LOG_DIR = Path(CONFIG["paths"]["output_root"]) / "corruption_logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    main()
