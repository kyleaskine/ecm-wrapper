# ECM Distributed Client

Standalone Python client for distributed integer factorization using GMP-ECM and YAFU.

## Quick Start

1. Install dependencies:
   ```bash
   pip install requests pyyaml
   ```

2. Configure your client in `client.yaml`

3. Run factorization:
   ```bash
   # ECM with GMP-ECM
   python3 ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000
   
   # P-1 with YAFU  
   python3 yafu-wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000
   ```

4. Batch processing:
   ```bash
   # Run from client/ directory
   scripts/run_pm1_batch_yafu.sh
   ```

## Files

- `ecm-wrapper.py` - GMP-ECM client wrapper
- `yafu-wrapper.py` - YAFU client wrapper
- `client.yaml` - Configuration file
- `scripts/` - Batch processing scripts
- `data/` - All outputs and data files:
  - `data/logs/` - Client logs
  - `data/outputs/` - Raw program outputs
  - `data/factors_found.txt` - Discovered factors log
  - `data/residues/` - ECM residue files
  - `data/results/` - Failed API submissions

## Configuration

Edit `client.yaml` to set:
- API endpoints 
- Binary paths for ecm/yafu
- Client identification
- Default parameters

See `CLAUDE.md` for detailed documentation.