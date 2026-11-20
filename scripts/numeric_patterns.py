"""
scripts/numeric_patterns.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import re
from typing import List, Dict, Any


UNITS = (
    r"mg/kg/day|mg/kg|mg/day|mg/d|"
    r"µg/kg|μg/kg|ug/kg|"
    r"ng/mL|ng/ml|"
    r"pg/mL|pg/ml|"
    r"mg/mL|mg/ml|"
    r"mg/L|"
    r"g/L|"
    r"mmol/L|mmol/l|"
    r"µmol/L|µmol/l|umol/L|umol/l|"
    r"mol/L|mol/l|"
    r"IU/L|IU/l|U/L|U/l|"

    r"cm/s|cm/sec|"
    r"m/s|m/sec|"
    r"GBq/kg|MBq/kg|kBq/kg|Bq/kg|"
    r"mCi/kg|µCi/kg|μCi/kg|uCi/kg|"

    r"mL/min|ml/min|mL/s|ml/s|"

    r"mmHg|"

    r"mGy|mSv|Gy|Sv|"
    r"GBq|MBq|kBq|Bq|"
    r"mCi|µCi|μCi|uCi|"

    r"HU|"

    r"milliseconds?|msecs?|ms|"
    r"seconds?|secs?|"
    r"minutes?|mins?|"
    r"hours?|hrs?|"
    r"days?|"
    r"weeks?|wks?|"
    r"months?|mos?|"
    r"years?|yrs?|"

    r"MHz|kHz|Hz|ppm|"

    r"mm|cm|mg|kg|g|mL|ml|µg|μg|ug|L|"

    r"bar|psi"
)


PATTERNS = [

    {
        "name": "epidemiology_in_range",
        "priority": 115,
        "regex": re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?P<numerator>\d+(?:\.\d+)?)"
            r"(?P<ratio_word>\s+in\s+)"
            r"(?P<denominator1>\d[\d,]*(?:\.\d+)?)"
            r"(?P<separator>\s*[-–]\s*)"
            r"(?P<denominator2>\d[\d,]*(?:\.\d+)?)(?!\s*%)",
            flags=re.IGNORECASE,
        ),
    },
    {
        "name": "epidemiology_numeric_ratio",
        "priority": 110,
        "regex": re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?P<numerator>\d+(?:\.\d+)?)"
            r"(?P<ratio_separator>\s*:\s*|\s+(?:per|in)\s+)"
            r"(?P<denominator>\d[\d,]*(?:\.\d+)?)(?!\s*%)",
            flags=re.IGNORECASE,
        ),
    },
    {
        "name": "epidemiology_rate_range",
        "priority": 112,
        "regex": re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?P<value1>\d+(?:\.\d+)?)"
            r"(?P<separator>\s*[-–]\s*)"
            r"(?P<value2>\d+(?:\.\d+)?)"
            r"(?P<space>\s+)"
            r"(?P<unit>per\s+(?:million|thousand|"
            r"(?:\d{1,3}(?:,\d{3})+|\d+)))\b",
            flags=re.IGNORECASE,
        ),
    },
    {
        "name": "epidemiology_per_word",
        "priority": 105,
        "regex": re.compile(
            r"(?P<value>\d+(?:\.\d+)?)"
            r"(?P<suffix>\s+per\s+(?:million|thousand)\b)",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "unitless_threshold_range",
        "priority": 65,
        "regex": re.compile(
            r"(?P<prefix>(?:<=|>=|≤|≥|<|>)\s*)?"
            r"(?P<value1>\d+(?:\.\d+)?)"
            r"(?P<separator>\s*[-–]\s*|\s+to\s+)"
            r"(?P<value2>\d+(?:\.\d+)?)"
            r"(?!\s*,\s*\d)"
        ),
    },
    {
        "name": "unitless_threshold",
        "priority": 60,
        "regex": re.compile(
            r"(?P<prefix>(?:<=|>=|≤|≥|<|>)\s*)"
            r"(?P<value>\d+(?:\.\d+)?)"
        ),
    },


    {
        "name": "range_with_unit",
        "priority": 100,
        "regex": re.compile(
            rf"(?P<value1>\d+(?:\.\d+)?)"
            rf"(?P<separator>\s*[-–]\s*)"
            rf"(?P<value2>\d+(?:\.\d+)?)"
            rf"(?P<space>\s*)"
            rf"(?P<unit>{UNITS})"
            rf"\b",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "textual_range_with_unit",
        "priority": 100,
        "regex": re.compile(
            rf"(?P<value1>\d+(?:\.\d+)?)"
            rf"(?P<separator>\s+to\s+)"
            rf"(?P<value2>\d+(?:\.\d+)?)"
            rf"(?P<space>\s*)"
            rf"(?P<unit>{UNITS})"
            rf"\b",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "dimension_2d_with_unit",
        "priority": 95,
        "regex": re.compile(
            rf"(?P<value1>\d+(?:\.\d+)?)"
            rf"(?P<separator>\s*[x×]\s*)"
            rf"(?P<value2>\d+(?:\.\d+)?)"
            rf"(?P<space>\s*)"
            rf"(?P<unit>{UNITS})"
            rf"\b",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "percentage_range",
        "priority": 95,
        "regex": re.compile(
            r"(?P<value1>\d+(?:\.\d+)?)"
            r"(?P<separator>\s*(?:%\s*)?[-–]\s*)"
            r"(?P<value2>\d+(?:\.\d+)?)"
            r"(?P<space>\s*)"
            r"(?P<unit>%)"

        ),
    },
    {
        "name": "degree_range",
        "priority": 95,
        "regex": re.compile(
            r"(?P<value1>\d+(?:\.\d+)?)"
            r"(?P<separator>\s*[-–]\s*|\s+to\s+)"
            r"(?P<value2>\d+(?:\.\d+)?)"
            r"(?P<space>\s*)"
            r"(?P<unit>°|degrees?)",
            flags=re.IGNORECASE,
        ),
    },
    {
        "name": "degree",
        "priority": 80,
        "regex": re.compile(
            r"(?P<value>\d+(?:\.\d+)?)"
            r"(?P<space>\s*)"
            r"(?P<unit>°|degrees?)",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "hyphenated_age",
        "priority": 90,
        "regex": re.compile(
            r"(?P<value>\d+(?:\.\d+)?)"
            r"(?P<space>-)"
            r"(?P<unit>year-old|years-old)"
            r"\b",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "age",
        "priority": 85,
        "regex": re.compile(
            r"(?P<value>\d+(?:\.\d+)?)"
            r"(?P<space>\s+)"
            r"(?P<unit>years?|yrs?)"
            r"\b",
            flags=re.IGNORECASE,
        ),
    },

    {
        "name": "percentage",
        "priority": 80,
        "regex": re.compile(
            r"(?P<value>\d+(?:\.\d+)?)"
            r"(?P<space>\s*)"
            r"(?P<unit>%)"
        ),
    },

    {
        "name": "value_with_unit",
        "priority": 70,
        "regex": re.compile(
            rf"(?P<value>\d+(?:\.\d+)?)"
            rf"(?P<space>\s*)"
            rf"(?P<unit>{UNITS})"
            rf"\b",
            flags=re.IGNORECASE,
        ),
    },
]


EPIDEMIOLOGY_CUES = re.compile(
    r"\b(?:prevalence|incidence|risk|frequency|occurs?|occurrence|"
    r"live births?|population|people|patients?|cases?)\b",
    flags=re.IGNORECASE,
)

THRESHOLD_CUES = re.compile(
    r"\b(?:cut-?off|threshold|normal|abnormal|ratio|index|score|"
    r"criterion|criteria|sensitivity|specificity|"
    r"resistive index|pulsatility index|RI|PI)\b",
    flags=re.IGNORECASE,
)

THRESHOLD_RANGE_CUES = re.compile(
    r"(?:cut-?off|threshold|normal(?:\s+range|\s+ratio)?|"
    r"abnormal(?:\s+range|\s+ratio)?|reference\s+range|"
    r"ratio|index|resistive\s+index|pulsatility\s+index|RI|PI)"
    r"(?:\s+(?:is|of|=|was|are|between|from))?\s*$",
    flags=re.IGNORECASE,
)


def has_context_cue(
    text: str,
    start: int,
    end: int,
    cue_regex: re.Pattern,
    window: int = 80,
) -> bool:
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)

    context = text[context_start:context_end]

    return cue_regex.search(context) is not None


def has_preceding_cue(
    text: str,
    start: int,
    cue_regex: re.Pattern,
    window: int = 40,
) -> bool:
    context_start = max(0, start - window)
    left_context = text[context_start:start]
    return cue_regex.search(left_context) is not None


def collect_candidates(text: str) -> List[Dict[str, Any]]:

    candidates = []

    for pattern_info in PATTERNS:

        pattern_name = pattern_info["name"]
        priority = pattern_info["priority"]
        regex = pattern_info["regex"]

        for match in regex.finditer(text):

            if pattern_name.startswith("epidemiology_"):
                if not has_context_cue(
                    text,
                    match.start(),
                    match.end(),
                    EPIDEMIOLOGY_CUES,
                ):
                    continue

            if pattern_name == "unitless_threshold":
                if not has_context_cue(
                    text,
                    match.start(),
                    match.end(),
                    THRESHOLD_CUES,
                    window=60,
                ):
                    continue

            if pattern_name == "unitless_threshold_range":
                if not has_preceding_cue(
                    text,
                    match.start(),
                    THRESHOLD_RANGE_CUES,
                    window=40,
                ):
                    continue

            candidates.append({
                "pattern_type": pattern_name,
                "priority": priority,

                "start": match.start(),
                "end": match.end(),

                "original_text": match.group(0),

                "groups": match.groupdict(),
            })

    return candidates


def spans_overlap(
    start1: int,
    end1: int,
    start2: int,
    end2: int,
) -> bool:

    return (
        start1 < end2
        and start2 < end1
    )


def remove_overlaps(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -item["priority"],
            -(item["end"] - item["start"]),
            item["start"],
        )
    )

    accepted = []

    for candidate in sorted_candidates:

        overlap_found = False

        for existing in accepted:

            if spans_overlap(
                candidate["start"],
                candidate["end"],
                existing["start"],
                existing["end"],
            ):
                overlap_found = True
                break

        if not overlap_found:
            accepted.append(candidate)

    accepted.sort(
        key=lambda item: item["start"]
    )

    return accepted


def detect_clinical_numbers(
    text: str
) -> List[Dict[str, Any]]:

    candidates = collect_candidates(text)

    candidates = remove_overlaps(
        candidates
    )

    return candidates
