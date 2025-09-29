#!/bin/bash
while IFS= read -r number <&3; do
    echo "Starting ECM on: $number"
    python3 ecm-wrapper.py -n "$number" --two-stage -v --b1 11000000 --b2 0
    echo "Completed: $number"
    echo "---"
done 3< data/numbers.txt