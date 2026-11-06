#!/usr/bin/env python3
"""
ECM Client Setup Script

Interactive setup wizard to create client.local.yaml configuration file.
Run this before using ecm_client.py for the first time, or re-run to
change your settings (existing values are used as defaults).

Usage:
    python3 setup.py
"""

import sys

if sys.version_info < (3, 9):
    print(f"Error: Python 3.9+ is required (you have Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    sys.exit(1)

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def get_input(prompt: str, default: str = "", required: bool = False) -> str:
    """Get user input with optional default value."""
    if default:
        display_prompt = f"{prompt} [{default}]: "
    else:
        display_prompt = f"{prompt}: "

    while True:
        value = input(display_prompt).strip()
        if not value:
            if default:
                return default
            elif required:
                print("  This field is required. Please enter a value.")
                continue
            else:
                return ""
        return value


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print("  Please enter 'y' or 'n'")


def detect_cpu_cores() -> int:
    """Detect number of CPU cores."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def detect_total_ram_gb() -> Optional[float]:
    """Detect total system RAM in GB. Returns None if detection fails."""
    try:
        if sys.platform == 'linux':
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        # MemTotal is in kB
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        elif sys.platform == 'darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return None


# Approximate per-worker peak memory for GMP-ECM stage 2 at various B1 levels
# (with GMP-ECM's default B2). These are conservative estimates.
_B1_MEMORY_TABLE = [
    # (max_b1, approx GB per worker)
    (11_000_000,    0.3),
    (43_000_000,    0.7),
    (110_000_000,   1.5),
    (260_000_000,   3.0),
    (850_000_000,   6.0),
    (2_900_000_000, 15.0),
]


def suggest_max_b1(total_ram_gb: float, workers: int) -> Optional[int]:
    """Suggest a max B1 value based on available RAM and worker count.

    Reserves ~2 GB for the OS and other processes, then divides the rest
    across workers. Returns the highest B1 that fits, or None if even the
    largest B1 fits comfortably.
    """
    usable_gb = max(total_ram_gb - 2.0, 1.0)
    per_worker_gb = usable_gb / workers

    best_b1: Optional[int] = None
    for b1, mem_gb in _B1_MEMORY_TABLE:
        if per_worker_gb >= mem_gb:
            best_b1 = b1
        else:
            break

    # If every entry fits, no limit needed
    if best_b1 == _B1_MEMORY_TABLE[-1][0]:
        return None
    return best_b1


def detect_hostname() -> str:
    """Detect machine hostname."""
    try:
        return platform.node() or "my-machine"
    except Exception:
        return "my-machine"


def find_binary(name: str, common_paths: list) -> str:
    """Try to find a binary in PATH or common locations."""
    # Check PATH first
    found = shutil.which(name)
    if found:
        return found

    # Check common paths
    for path in common_paths:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    return ""


def check_gpu() -> tuple:
    """Check for NVIDIA GPU availability."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            return True, gpu_name
    except Exception:
        pass
    return False, None


def load_existing_config() -> Dict[str, Any]:
    """Load existing client.local.yaml if present, return as nested dict."""
    config_path = Path("client.local.yaml")
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested config value."""
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def verify_ecm_installation() -> bool:
    """Run a quick ECM test via ecm_wrapper to verify the full pipeline works."""
    # 85643 = 131 x 653 -- small enough to factor instantly at B1=1000
    test_composite = "85643"
    script_dir = Path(__file__).parent

    try:
        result = subprocess.run(
            [sys.executable, str(script_dir / "ecm_wrapper.py"),
             "--composite", test_composite, "--b1", "1000", "--curves", "5"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_dir)
        )

        output = result.stdout + result.stderr

        if "Factor found" in output or "factors_found" in output:
            print("  ECM found a factor -- everything is working!")
            return True
        elif result.returncode == 0:
            print("  ECM ran successfully (no factor found, but the pipeline works)")
            return True
        else:
            print("  WARNING: ecm_wrapper exited with an error:")
            # Show last few relevant lines
            for line in output.strip().split('\n')[-5:]:
                print(f"  {line}")
            return False

    except FileNotFoundError:
        print("  ERROR: Could not run ecm_wrapper.py")
        return False
    except subprocess.TimeoutExpired:
        print("  WARNING: ECM test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def format_b1_human(b1: Optional[int]) -> str:
    """Format a B1 value for display (e.g., 110000000 -> '110M')."""
    if b1 is None:
        return "none"
    if b1 >= 1_000_000_000:
        return f"{b1 / 1_000_000_000:.0f}B"
    if b1 >= 1_000_000:
        return f"{b1 / 1_000_000:.0f}M"
    if b1 >= 1_000:
        return f"{b1 / 1_000:.0f}K"
    return str(b1)


def yaml_quote_path(path: str) -> str:
    """Quote a path for YAML if it contains special characters."""
    if " " in path or path.startswith("~"):
        return f'"{path}"'
    return path


def main():
    print()
    print("=" * 60)
    print("  ECM Client Setup Wizard")
    print("=" * 60)
    print()

    # Load existing config for defaults
    config_path = Path("client.local.yaml")
    existing = load_existing_config()

    if existing:
        print("Existing client.local.yaml found. Your current values will")
        print("be shown as defaults -- press Enter to keep them.")
        print()
    else:
        print("This wizard will help you create a client.local.yaml file")
        print("with your personal settings for the ECM factorization client.")
        print()

    # ============================================================
    # User Information
    # ============================================================
    print("-" * 60)
    print("USER INFORMATION")
    print("-" * 60)
    print()

    existing_username = get_nested(existing, 'client', 'username', default="")
    username = get_input(
        "Enter your username (for tracking your contributions)",
        default=existing_username,
        required=True
    )

    existing_machine = get_nested(existing, 'client', 'cpu_name', default="")
    default_machine = existing_machine or detect_hostname()
    machine_name = get_input(
        "Enter a name for this machine",
        default=default_machine
    )

    # ============================================================
    # API Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("API CONFIGURATION")
    print("-" * 60)
    print()

    existing_endpoint = get_nested(existing, 'api', 'endpoint', default="")

    print("The ECM client submits results to a coordination server.")
    print("The default production server is: https://ecm.kyleaskine.com/api/v1")
    print()

    if existing_endpoint:
        api_endpoint = get_input("API endpoint URL", default=existing_endpoint)
    else:
        use_default_api = get_yes_no("Use the default production server?", default=True)
        if use_default_api:
            api_endpoint = "https://ecm.kyleaskine.com/api/v1"
        else:
            api_endpoint = get_input(
                "Enter API endpoint URL",
                default="http://localhost:8000/api/v1"
            )

    # ============================================================
    # GPU Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("GPU CONFIGURATION")
    print("-" * 60)
    print()

    has_gpu, gpu_name = check_gpu()
    existing_gpu = get_nested(existing, 'programs', 'gmp_ecm', 'gpu_enabled', default=None)

    if has_gpu:
        print(f"Detected GPU: {gpu_name}")
        gpu_default = existing_gpu if existing_gpu is not None else True
        gpu_enabled = get_yes_no("Enable GPU acceleration?", default=gpu_default)
    else:
        print("No NVIDIA GPU detected.")
        gpu_default = existing_gpu if existing_gpu is not None else False
        gpu_enabled = get_yes_no("Enable GPU anyway? (for manual setup)", default=gpu_default)

    existing_gpu_device = get_nested(existing, 'programs', 'gmp_ecm', 'gpu_device', default=0)
    gpu_device = 0
    if gpu_enabled:
        gpu_device_str = get_input("GPU device number", default=str(existing_gpu_device))
        try:
            gpu_device = int(gpu_device_str)
        except ValueError:
            gpu_device = 0

    # ============================================================
    # ECM Binary
    # ============================================================
    print()
    print("-" * 60)
    print("ECM BINARY")
    print("-" * 60)
    print()

    existing_ecm_path = get_nested(existing, 'programs', 'gmp_ecm', 'path', default="")

    # Try to find ECM binary
    ecm_paths = [
        "~/ecm",
        "~/ecm-master/ecm",
        "/usr/local/bin/ecm",
        "/usr/bin/ecm",
        "~/gmp-ecm/ecm",
    ]
    detected_ecm = find_binary("ecm", ecm_paths)

    if existing_ecm_path:
        ecm_path = get_input("Path to ECM binary", default=existing_ecm_path)
    elif detected_ecm:
        print(f"Detected ECM binary: {detected_ecm}")
        use_detected = get_yes_no("Use this ECM binary?", default=True)
        if use_detected:
            ecm_path = detected_ecm
        else:
            ecm_path = get_input("Enter path to ECM binary", required=True)
    else:
        print("ECM binary not found in common locations.")
        print("You can download pre-built binaries from:")
        print("  https://ecm.kyleaskine.com/downloads/")
        print()
        ecm_path = get_input(
            "Enter path to ECM binary (or 'ecm' if in PATH)",
            default="ecm"
        )

    # ============================================================
    # Worker Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("WORKER CONFIGURATION")
    print("-" * 60)
    print()

    cpu_cores = detect_cpu_cores()
    print(f"Detected {cpu_cores} CPU cores.")

    existing_workers = get_nested(existing, 'programs', 'gmp_ecm', 'workers', default=None)
    default_workers = existing_workers if existing_workers is not None else min(cpu_cores, 8)
    workers_str = get_input(
        "Number of parallel workers for stage 2 / multiprocess",
        default=str(default_workers)
    )
    try:
        workers = int(workers_str)
    except ValueError:
        workers = default_workers

    # ============================================================
    # Stage 2 Memory Limit (max B1 for residues)
    # ============================================================
    print()
    print("-" * 60)
    print("STAGE 2 MEMORY LIMIT")
    print("-" * 60)
    print()

    print("Each stage 2 worker uses memory that scales with B1.")
    print(f"With {workers} workers running in parallel, the total")
    print("memory needed is roughly workers x per-worker usage.")
    print()

    total_ram = detect_total_ram_gb()
    suggested = None
    existing_max_b1 = get_nested(existing, 'programs', 'gmp_ecm', 'stage2_max_b1', default=None)

    if total_ram:
        print(f"Detected {total_ram:.0f} GB RAM.")
        suggested = suggest_max_b1(total_ram, workers)
        if suggested:
            print(f"Recommended max B1 for {workers} workers: {format_b1_human(suggested)} ({suggested:,})")
        else:
            print(f"Your system has plenty of RAM for {workers} workers -- no limit needed.")
    else:
        print("Could not detect system RAM. Reference table")
        print(f"(for {workers} workers running simultaneously):")
        print()
        # Show a dynamic table based on worker count
        for label_gb in [8, 16, 32, 64]:
            rec = suggest_max_b1(label_gb, workers)
            if rec:
                print(f"  {label_gb:>3} GB RAM  ->  stage2_max_b1: {rec:<13,} ({format_b1_human(rec)})")
            else:
                print(f"  {label_gb:>3} GB RAM  ->  no limit needed")
    print()

    # Determine default for the prompt
    if existing_max_b1:
        default_max_b1_display = str(existing_max_b1)
    elif suggested:
        default_max_b1_display = str(suggested)
    else:
        default_max_b1_display = ""

    max_b1_str = get_input(
        "Max B1 for stage 2 residues (Enter for no limit)",
        default=default_max_b1_display
    )

    stage2_max_b1: Optional[int] = None
    if max_b1_str:
        try:
            stage2_max_b1 = int(float(max_b1_str))
        except ValueError:
            print(f"  Could not parse '{max_b1_str}', using no limit")
            stage2_max_b1 = None

    if stage2_max_b1:
        print(f"  -> Will only accept stage 2 residues with B1 <= {format_b1_human(stage2_max_b1)}")
    else:
        print("  -> No limit (will accept all stage 2 residues)")

    # ============================================================
    # Optional: YAFU Binary
    # ============================================================
    print()
    print("-" * 60)
    print("OPTIONAL: YAFU BINARY")
    print("-" * 60)
    print()

    existing_yafu_path = get_nested(existing, 'programs', 'yafu', 'path', default=None)
    existing_yafu_threads = get_nested(existing, 'programs', 'yafu', 'threads', default=None)

    if existing_yafu_path:
        configure_yafu = get_yes_no("Keep YAFU configured?", default=True)
    else:
        configure_yafu = get_yes_no("Do you have YAFU installed?", default=False)

    yafu_path = None
    yafu_threads = workers

    if configure_yafu:
        yafu_paths = [
            "~/yafu/yafu",
            "~/yafu-master/yafu",
            "/usr/local/bin/yafu",
        ]
        detected_yafu = find_binary("yafu", yafu_paths)

        yafu_default = existing_yafu_path or detected_yafu or ""
        if yafu_default and not existing_yafu_path:
            print(f"Detected YAFU binary: {yafu_default}")
        yafu_path = get_input("Path to YAFU", default=yafu_default)

        if yafu_path:
            default_yafu_threads = existing_yafu_threads if existing_yafu_threads is not None else workers
            yafu_threads_str = get_input("YAFU threads", default=str(default_yafu_threads))
            try:
                yafu_threads = int(yafu_threads_str)
            except ValueError:
                yafu_threads = workers

    # ============================================================
    # Generate Configuration
    # ============================================================
    print()
    print("-" * 60)
    print("GENERATING CONFIGURATION")
    print("-" * 60)
    print()

    # Build the YAML content
    config_lines = [
        "# ECM Client Local Configuration",
        f"# Generated by setup.py on {platform.node()}",
        "# This file overrides settings in client.yaml",
        "",
        "api:",
        f'  endpoint: "{api_endpoint}"',
        "",
        "client:",
        f'  username: "{username}"',
        f'  cpu_name: "{machine_name}"',
        "",
        "programs:",
        "  gmp_ecm:",
        f"    path: {yaml_quote_path(ecm_path)}",
        f"    gpu_enabled: {'true' if gpu_enabled else 'false'}",
        f"    gpu_device: {gpu_device}",
        f"    workers: {workers}",
    ]

    if stage2_max_b1 is not None:
        config_lines.append(f"    stage2_max_b1: {stage2_max_b1}")

    # Optional YAFU config
    if yafu_path:
        config_lines.extend([
            "",
            "  yafu:",
            f"    path: {yaml_quote_path(yafu_path)}",
            f"    threads: {yafu_threads}",
        ])

    config_lines.append("")  # Trailing newline

    config_content = "\n".join(config_lines)

    # Show preview
    print("Configuration preview:")
    print()
    print("-" * 40)
    print(config_content)
    print("-" * 40)
    print()

    if get_yes_no("Save this configuration?", default=True):
        with open(config_path, 'w') as f:
            f.write(config_content)

        print()
        print("=" * 60)
        print("  SETUP COMPLETE!")
        print("=" * 60)
        print()
        print(f"Configuration saved to: {config_path.absolute()}")
        print()
        print("You can now run the ECM client:")
        print()
        print("  # Auto-work mode (get work from server)")
        print("  python3 ecm_client.py")
        print()
        print("  # With specific parameters")
        if gpu_enabled:
            print("  python3 ecm_client.py --b1 11000000 --stage1-only   # GPU stage 1")
        else:
            print("  python3 ecm_client.py --b1 11000000 --multiprocess")
        print()
        print("  # Test without submitting (ecm_wrapper.py)")
        print("  python3 ecm_wrapper.py --composite \"123456789\" --curves 10 --b1 11000")
        print()
        print("To change your settings later, just run setup.py again.")
        print()
        print("For more options, run: python3 ecm_client.py --help")
        print()

        # Offer verification test
        if get_yes_no("Run a quick test to verify ECM is working?", default=True):
            print()
            print("Running ECM verification (factoring a small test number)...")
            if verify_ecm_installation():
                print()
                print("Setup and verification complete! You're ready to go.")
            else:
                print()
                print("ECM verification failed. Please check your ECM binary path")
                print(f"and that client.local.yaml is correct.")
            print()
    else:
        print("\nSetup cancelled. No files were written.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
