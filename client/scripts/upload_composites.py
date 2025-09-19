#!/usr/bin/env python3
"""
Upload composites from o21.csv to the ECM server.

This script reads the o21.csv file and uploads the composite numbers to the server
using the bulk upload API. It handles duplicates automatically by skipping them.
"""

import csv
import requests
import sys
import argparse
from typing import List, Dict, Any

def read_o21_csv(filename: str) -> List[str]:
    """
    Read composite numbers from the o21.csv file.

    Args:
        filename: Path to the CSV file

    Returns:
        List of composite numbers as strings
    """
    composites = []

    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Extract the composite number from SmallestComposite column
                composite = row.get('SmallestComposite', '').strip()
                if composite and composite.isdigit():
                    composites.append(composite)
                    print(f"Found composite: {composite[:50]}{'...' if len(composite) > 50 else ''} ({len(composite)} digits)")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    return composites

def upload_composites(composites: List[str], server_url: str, batch_size: int = 100) -> None:
    """
    Upload composites to the server in batches.

    Args:
        composites: List of composite numbers to upload
        server_url: Base URL of the ECM server
        batch_size: Number of composites to upload per batch
    """
    url = f"{server_url}/api/v1/admin/composites/bulk"

    total_new = 0
    total_existing = 0
    total_errors = 0

    # Process in batches
    for i in range(0, len(composites), batch_size):
        batch = composites[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(composites) + batch_size - 1) // batch_size

        print(f"\nUploading batch {batch_num}/{total_batches} ({len(batch)} composites)...")

        try:
            response = requests.post(
                url,
                json=batch,  # Send the list directly, not wrapped in an object
                params={"default_priority": 0},  # Send priority as query parameter
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                new_count = result.get('new_composites', 0)
                existing_count = result.get('existing_composites', 0)
                invalid_count = result.get('invalid_numbers', 0)

                total_new += new_count
                total_existing += existing_count
                total_errors += invalid_count

                print(f"  ✓ New: {new_count}, Existing: {existing_count}, Invalid: {invalid_count}")

                if result.get('errors'):
                    print(f"  Errors: {result['errors']}")

            else:
                print(f"  ✗ HTTP {response.status_code}: {response.text}")
                total_errors += len(batch)

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Network error: {e}")
            total_errors += len(batch)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            total_errors += len(batch)

    # Summary
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {len(composites)}")
    print(f"New composites added: {total_new}")
    print(f"Already existed (skipped): {total_existing}")
    print(f"Errors/Invalid: {total_errors}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Upload composites from o21.csv to ECM server")
    parser.add_argument("--csv", default="o21.csv", help="Path to CSV file (default: o21.csv)")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for uploads (default: 100)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")

    args = parser.parse_args()

    print(f"Reading composites from: {args.csv}")
    composites = read_o21_csv(args.csv)

    print(f"\nFound {len(composites)} composite numbers")

    if args.dry_run:
        print("\nDRY RUN - Would upload these composites:")
        for i, comp in enumerate(composites[:10]):  # Show first 10
            print(f"  {i+1}: {comp[:50]}{'...' if len(comp) > 50 else ''} ({len(comp)} digits)")
        if len(composites) > 10:
            print(f"  ... and {len(composites) - 10} more")
        print(f"\nTo actually upload, run without --dry-run")
        return

    print(f"Server: {args.server}")
    print(f"Batch size: {args.batch_size}")

    # Test server connectivity
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code != 200:
            print(f"Warning: Server health check failed (HTTP {response.status_code})")
    except Exception as e:
        print(f"Warning: Cannot connect to server: {e}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Confirm upload
    response = input(f"\nUpload {len(composites)} composites to {args.server}? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled")
        sys.exit(0)

    upload_composites(composites, args.server, args.batch_size)

if __name__ == "__main__":
    main()