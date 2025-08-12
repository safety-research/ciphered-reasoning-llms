#!/usr/bin/env python3
"""
count_tokens_by_section.py

Counts the number of tokens between ## markdown headers in a text file using tiktoken.
Provides statistics about section lengths and can optionally output a CSV report.

Usage:
  python count_tokens_by_section.py input.txt
  python count_tokens_by_section.py input.txt --encoding cl100k_base
  python count_tokens_by_section.py input.txt --csv output.csv
  python count_tokens_by_section.py input.txt --verbose
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import csv

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is not installed. Install with: pip install tiktoken", file=sys.stderr)
    sys.exit(1)


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


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using specified tiktoken encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(description="Count tokens between ## markdown headers")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument(
        "--encoding", default="cl100k_base", help="Tiktoken encoding to use (default: cl100k_base for GPT-3.5/GPT-4)"
    )
    parser.add_argument("--csv", help="Output results to CSV file")
    parser.add_argument("--verbose", action="store_true", help="Show content preview for each section")
    parser.add_argument("--min-tokens", type=int, default=0, help="Only show sections with at least this many tokens")
    parser.add_argument(
        "--max-tokens", type=int, default=float("inf"), help="Only show sections with at most this many tokens"
    )

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        text = input_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract sections
    sections = extract_sections(text)

    if not sections:
        print("No sections found in the file.")
        sys.exit(0)

    # Count tokens for each section
    results = []
    total_tokens = 0

    print(f"\nAnalyzing {len(sections)} sections using encoding: {args.encoding}\n")
    print("=" * 80)

    for header, content in sections:
        if not content:  # Skip empty sections
            continue

        token_count = count_tokens(content, args.encoding)
        total_tokens += token_count

        # Apply filters
        if token_count < args.min_tokens or token_count > args.max_tokens:
            continue

        results.append({"header": header, "tokens": token_count, "characters": len(content), "content": content})

    # Sort by token count (descending)
    results.sort(key=lambda x: x["tokens"], reverse=True)

    # Display results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['header']}")
        print(f"   Tokens: {format_number(result['tokens'])}")
        print(f"   Characters: {format_number(result['characters'])}")

        if args.verbose:
            # Show first 200 characters of content
            preview = result["content"][:200].replace("\n", " ")
            if len(result["content"]) > 200:
                preview += "..."
            print(f"   Preview: {preview}")

        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if results:
        token_counts = [r["tokens"] for r in results]
        avg_tokens = sum(token_counts) / len(token_counts)

        print(f"Total sections: {len(results)}")
        print(f"Total tokens: {format_number(total_tokens)}")
        print(f"Average tokens per section: {format_number(int(avg_tokens))}")
        print(f"Median tokens per section: {format_number(sorted(token_counts)[len(token_counts)//2])}")
        print(f"Smallest section: {format_number(min(token_counts))} tokens")
        print(f"Largest section: {format_number(max(token_counts))} tokens")

        # Show top 5 largest sections
        print("\nTop 5 largest sections:")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. {result['header']}: {format_number(result['tokens'])} tokens")

    # Output to CSV if requested
    if args.csv:
        csv_path = Path(args.csv)
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["header", "tokens", "characters"])
                writer.writeheader()
                for result in results:
                    writer.writerow(
                        {"header": result["header"], "tokens": result["tokens"], "characters": result["characters"]}
                    )
            print(f"\nCSV report saved to: {csv_path}")
        except Exception as e:
            print(f"Error writing CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
