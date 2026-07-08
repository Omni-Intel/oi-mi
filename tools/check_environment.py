"""Check the local Python environment for oi-mi.

This script intentionally uses only the Python standard library so it can run
before project dependencies are installed.
"""

import importlib.util
import platform
import subprocess
import sys


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 12

RUNTIME_MODULES = (
    ("bc-ecap-sdk", "bc_ecap_sdk"),
    ("brainflow", "brainflow"),
    ("braindecode", "braindecode"),
    ("click", "click"),
    ("mne", "mne"),
    ("moabb", "moabb"),
    ("numpy", "numpy"),
    ("pyedflib", "pyedflib"),
    ("pylsl", "pylsl"),
    ("pyriemann", "pyriemann"),
    ("pyserial", "serial"),
    ("pyyaml", "yaml"),
    ("rich", "rich"),
    ("scikit-learn", "sklearn"),
    ("scipy", "scipy"),
    ("streamlit", "streamlit"),
    ("torch", "torch"),
    ("zeroconf", "zeroconf"),
)


def is_virtual_environment():
    return (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or hasattr(sys, "real_prefix")
    )


def pip_version():
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except Exception as exc:
        return "unavailable: %s" % exc
    return completed.stdout.strip()


def missing_runtime_modules():
    missing = []
    for package_name, module_name in RUNTIME_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def suggested_commands():
    system = platform.system().lower()
    if system == "windows":
        return (
            "py -3.12 -m venv .venv",
            "source .venv/Scripts/activate",
            "python -m pip install -U pip setuptools wheel",
            "pip install -e .",
            "python tools/check_environment.py",
        )
    return (
        "python3.12 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install -U pip setuptools wheel",
        "pip install -e .",
        "python tools/check_environment.py",
    )


def main():
    errors = []
    warnings = []

    print("oi-mi environment check")
    print("platform: %s" % platform.platform())
    print("python: %s" % sys.version.replace("\n", " "))
    print("executable: %s" % sys.executable)
    print("virtualenv: %s" % ("yes" if is_virtual_environment() else "no"))
    print("pip: %s" % pip_version())

    version = sys.version_info
    if (version.major, version.minor) != (REQUIRED_MAJOR, REQUIRED_MINOR):
        errors.append(
            "Python %s.%s.x is required; current interpreter is %s.%s.%s."
            % (
                REQUIRED_MAJOR,
                REQUIRED_MINOR,
                version.major,
                version.minor,
                version.micro,
            )
        )

    if not is_virtual_environment():
        warnings.append("No virtual environment is active.")

    missing = missing_runtime_modules()
    if missing:
        errors.append(
            "Missing runtime dependencies: %s. Run `pip install -e .` inside the virtual environment."
            % ", ".join(missing)
        )

    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print("- %s" % warning)

    if errors:
        print("")
        print("Problems:")
        for error in errors:
            print("- %s" % error)
        print("")
        print("Suggested setup:")
        for command in suggested_commands():
            print("  %s" % command)
        return 1

    print("")
    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
