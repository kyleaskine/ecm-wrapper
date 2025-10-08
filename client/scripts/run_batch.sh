#!/bin/bash
echo "Starting ECM batch processing on $(wc -l < data/numbers.txt) numbers"
echo "=========================================="

count=0
total=$(wc -l < data/numbers.txt)

while IFS= read -r number <&3; do
    count=$((count + 1))
    echo "[$count/$total] Starting ECM on: $number"
    python3 ecm-wrapper.py -n "$number" --two-stage -v --b1 62000000 --b2 0
    echo "[$count/$total] Completed"
    echo "---"
done 3< data/numbers.txt

echo "ECM batch processing complete!"