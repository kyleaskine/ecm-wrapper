#!/bin/bash
echo "Starting P-1 batch testing on $(wc -l < data/numbers.txt) numbers"
echo "Using B1=1,000,000, B2=0 (Stage 1 only)"
echo "Results will be logged to data/factors_found.txt"
echo "=========================================="

count=0
total=$(wc -l < data/numbers.txt)

while IFS= read -r number <&3; do
    count=$((count + 1))
    echo "[$count/$total] Starting P-1 on $(echo $number | cut -c1-20)...$(echo $number | cut -c-20) ($(echo ${#number}) digits)"

    python3 ecm-wrapper.py -n "$number" --method pm1 -v

    echo "[$count/$total] Completed P-1 test"
    echo "---"
done 3< data/numbers.txt

echo "P-1 batch testing complete!"
echo "Check data/factors_found.txt for any discovered factors."