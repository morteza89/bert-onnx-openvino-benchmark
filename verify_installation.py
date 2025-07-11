#!/usr/bin/env python3
"""
Package Installation Verification Script
This script checks which packages from requirements-remaining.txt are actually installed
"""

import subprocess
import sys
import pkg_resources
import re


def get_installed_packages():
    """Get list of all installed packages"""
    installed = {}
    for dist in pkg_resources.working_set:
        installed[dist.project_name.lower()] = dist.version
    return installed


def parse_requirements_file(filename):
    """Parse requirements file and extract package names and versions"""
    requirements = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name and version
                    match = re.match(r'^([a-zA-Z0-9_-]+)==(.+)$', line)
                    if match:
                        pkg_name = match.group(1).lower()
                        version = match.group(2)
                        requirements[pkg_name] = version
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return {}
    return requirements


def main():
    print("=== Package Installation Verification ===")
    print()

    # Get installed packages
    installed = get_installed_packages()

    # Parse requirements file
    requirements = parse_requirements_file('requirements-remaining.txt')

    if not requirements:
        print("Could not parse requirements file")
        return

    print(
        f"Checking {len(requirements)} packages from requirements-remaining.txt:")
    print()

    missing_packages = []
    version_mismatches = []
    installed_packages = []

    for pkg_name, required_version in requirements.items():
        if pkg_name in installed:
            installed_version = installed[pkg_name]
            if installed_version == required_version:
                installed_packages.append((pkg_name, installed_version, "✓"))
                print(
                    f"✓ {pkg_name:<30} {installed_version} (matches {required_version})")
            else:
                version_mismatches.append(
                    (pkg_name, installed_version, required_version))
                print(
                    f"⚠ {pkg_name:<30} {installed_version} (expected {required_version})")
        else:
            missing_packages.append((pkg_name, required_version))
            print(f"✗ {pkg_name:<30} NOT INSTALLED (expected {required_version})")

    print()
    print("=== Summary ===")
    print(f"✓ Installed correctly: {len(installed_packages)}")
    print(f"⚠ Version mismatches: {len(version_mismatches)}")
    print(f"✗ Missing packages: {len(missing_packages)}")

    if missing_packages:
        print()
        print("=== Missing Packages ===")
        for pkg_name, version in missing_packages:
            print(f"  {pkg_name}=={version}")

        print()
        print("To install missing packages, run:")
        missing_list = [f"{pkg}=={ver}" for pkg, ver in missing_packages]
        print(f"pip install {' '.join(missing_list)}")

    if version_mismatches:
        print()
        print("=== Version Mismatches ===")
        for pkg_name, installed_ver, required_ver in version_mismatches:
            print(
                f"  {pkg_name}: installed {installed_ver}, required {required_ver}")


if __name__ == "__main__":
    main()
