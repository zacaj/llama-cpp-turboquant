#!/usr/bin/env python3
"""
Filter bench-results.tsv by regex patterns on any column, sorted by median tg/s.

Usage:
  bench-filter.py [-o OUTPUT] [COLUMN=REGEX ...]

Examples:
  # Find best config for Qwen3.6-27B model
  bench-filter.py '-m=Qwen3.6-27B'

  # Find turbo3 configs with Q8 KV cache
  bench-filter.py '--spec-draft-type-v=turbo3' '--spec-draft-type-k=q8'

  # Find p50.json results with high throughput
  bench-filter.py 'log_file=p50'

  # Output to file instead of stdout
  bench-filter.py -o results.tsv '-m=Agents' 'tg_median=[0-9]{2,}'
"""

import sys
import re
import csv
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Filter bench results by regex on any column',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: stdout)',
    )
    parser.add_argument(
        '-i', '--input',
        default='data/prompt-logger/bench-results.tsv',
        help='Input TSV file (default: data/prompt-logger/bench-results.tsv)',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Show only top N results by tg_median',
    )
    parser.add_argument(
        'filters',
        nargs='*',
        help='Column filters: COLUMN=REGEX (can specify multiple)',
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Parse filters
    column_filters = {}
    for filt in args.filters:
        if '=' not in filt:
            print(f"Error: filter must be in format COLUMN=REGEX: {filt}", file=sys.stderr)
            sys.exit(1)
        col, regex = filt.split('=', 1)
        try:
            column_filters[col] = re.compile(regex)
        except re.error as e:
            print(f"Error: invalid regex in {filt}: {e}", file=sys.stderr)
            sys.exit(1)

    # Read TSV
    rows = []
    headers = None
    col_indices = {}

    with open(input_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        headers = reader.fieldnames

        for row in reader:
            # Check all filters
            match = True
            for col, pattern in column_filters.items():
                if col not in row:
                    print(f"Error: column '{col}' not found in TSV", file=sys.stderr)
                    print(f"Available columns: {', '.join(headers)}", file=sys.stderr)
                    sys.exit(1)

                if not pattern.search(row[col]):
                    match = False
                    break

            if match:
                rows.append(row)

    # Sort by tg_median (descending)
    if 'tg_median' in headers:
        try:
            rows.sort(
                key=lambda r: float(r.get('tg_median', 0)),
                reverse=True
            )
        except ValueError:
            pass  # tg_median not numeric, skip sorting

    # Limit results
    if args.top:
        rows = rows[:args.top]

    # Write output
    output_file = sys.stdout
    if args.output:
        output_file = open(args.output, 'w')

    try:
        writer = csv.DictWriter(output_file, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output_file.close()

    print(f"\nFound {len(rows)} matching rows", file=sys.stderr)

if __name__ == '__main__':
    main()
