import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv
import os


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from markdown text, splitting on ## headers.
    Returns list of (header, content) tuples.
    """
    # Split on ## headers (but not ### or more)
    pattern = r"^### (.+?)$"

    sections = []
    current_header = "Beginning (no header)"
    current_content = []

    for line in text.split("\n"):
        match = re.match(pattern, line, re.MULTILINE)
        if match:
            # Save previous section if it has content
            if current_content:
                sections.append((current_header, "\n".join(current_content).strip()))
            # Start new section
            current_header = match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_content:
        sections.append((current_header, "\n".join(current_content).strip()))

    return sections


def get_designing_data_intensive_applications():
    with open(os.path.join(os.path.dirname(__file__), "raw/DDIA_cleaned.txt"), "r") as f:
        text = f.read()

    sections = extract_sections(text)

    return [section[1] for section in sections]
