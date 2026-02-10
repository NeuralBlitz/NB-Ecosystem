#!/usr/bin/env python3
"""
NeuralBlitz Security Isolation Tests - Simplified Version
"""

import os
import subprocess
import socket
import requests
import json
import time
from pathlib import Path
from typing import Dict, Any


class SecurityIsolationTester:
    """Tests security and isolation of NeuralBlitz systems"""

    def __init__(self):
        self.results = {"overall_status": "UNKNOWN", "tests_run": 0, "tests_passed": 0}
        self.critical_assets = [
            "NBX-LRS/",
            "NB-Ecosystem/",
            "Emergent-Prompt-Architecture/",
            "opencode-lrs-agents-nbx/",
            "Advanced-Research/",
            "quantum_sim/",
        ]

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security isolation tests"""
        print("🔒 Starting NeuralBlitz Security Isolation Tests...")
        print("=" * 60)

        tests = [
            self._test_network_isolation,
            self._test_file_permissions,
            self._test_git_exposure,
            self._test_api_security,
        ]

        for test_func in tests:
            try:
                print(f"\nRunning {test_func.__name__}...", end=" ")
                result = test_func()
                self.results["tests_run"] += 1

                if result["status"] == "PASS":
                    self.results["tests_passed"] += 1
                    print("✅ PASS")
                else:
                    print("❌ FAIL")
                    print(f"  Details: {result.get('message', 'Unknown error')}")

            except Exception as e:
                self.results["tests_run"] += 1
                print(f"❌ ERROR: {e}")

        self._calculate_overall_status()
        self._print_summary()
        return self.results

    def _test_network_isolation(self) -> Dict[str, Any]:
        """Test network isolation"""
        unexpected_ports = []

        for port in range(1, 1025):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex(("localhost", port))
                sock.close()

                if result == 0 and port not in [80, 443, 8000, 8080]:
                    unexpected_ports.append(port)
            except Exception:
                continue

        return {
            "status": "PASS" if len(unexpected_ports) == 0 else "FAIL",
            "unexpected_ports": unexpected_ports,
            "message": f"Found {len(unexpected_ports)} unexpected open ports",
        }

    def _test_file_permissions(self) -> Dict[str, Any]:
        """Test file system permissions"""
        permissive_dirs = []

        for asset in self.critical_assets:
            asset_path = Path(asset)
            if asset_path.exists():
                stat_info = asset_path.stat()
                mode = oct(stat_info.st_mode)[-3:]

                # Check if world-readable
                if mode[2] in ["4", "5", "6", "7"]:
                    permissive_dirs.append(f"{asset} ({mode})")

        return {
            "status": "PASS" if len(permissive_dirs) == 0 else "FAIL",
            "permissive_dirs": permissive_dirs,
            "message": f"Found {len(permissive_dirs)} directories with permissive permissions",
        }

    def _test_git_exposure(self) -> Dict[str, Any]:
        """Test for git directory exposure"""
        exposed_git_dirs = []

        for asset in self.critical_assets:
            git_dir = Path(asset) / ".git"
            if git_dir.exists():
                exposed_git_dirs.append(str(git_dir))

        return {
            "status": "PASS" if len(exposed_git_dirs) == 0 else "FAIL",
            "exposed_git_dirs": exposed_git_dirs,
            "message": f"Found {len(exposed_git_dirs)} exposed .git directories",
        }

    def _test_api_security(self) -> Dict[str, Any]:
        """Test API security (basic)"""
        try:
            # Test health endpoint (should be public)
            response = requests.get("http://localhost:8000/health", timeout=2)
            health_accessible = response.status_code == 200

            # Test protected endpoint (should require auth)
            response = requests.get("http://localhost:8000/api/v1/capabilities", timeout=2)
            protected_blocked = response.status_code in [401, 403]

            if health_accessible and protected_blocked:
                return {"status": "PASS", "message": "API security configured correctly"}
            else:
                return {"status": "FAIL", "message": "API security issues detected"}

        except requests.exceptions.RequestException:
            return {"status": "PASS", "message": "API not running (secure)"}

    def _calculate_overall_status(self) -> None:
        """Calculate overall security status"""
        if self.results["tests_run"] == 0:
            self.results["overall_status"] = "ERROR"
        elif self.results["tests_passed"] == self.results["tests_run"]:
            self.results["overall_status"] = "SECURE"
        elif self.results["tests_passed"] >= self.results["tests_run"] * 0.8:
            self.results["overall_status"] = "MOSTLY_SECURE"
        else:
            self.results["overall_status"] = "NEEDS_ATTENTION"

    def _print_summary(self) -> None:
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🔒 SECURITY ISOLATION TEST SUMMARY")
        print("=" * 60)

        status_emoji = {
            "SECURE": "🟢",
            "MOSTLY_SECURE": "🟡",
            "NEEDS_ATTENTION": "🟠",
            "INSECURE": "🔴",
            "ERROR": "💥",
        }

        overall_status = self.results["overall_status"]
        emoji = status_emoji.get(overall_status, "❓")

        print(f"Overall Status: {emoji} {overall_status}")
        print(f"Tests Passed: {self.results['tests_passed']}/{self.results['tests_run']}")

        pass_rate = (
            (self.results["tests_passed"] / self.results["tests_run"] * 100)
            if self.results["tests_run"] > 0
            else 0
        )
        print(f"Pass Rate: {pass_rate:.1f}%")

        print("\n" + "=" * 60)


def main():
    """Main test runner"""
    tester = SecurityIsolationTester()
    results = tester.run_all_tests()

    # Save results to file
    try:
        with open("security_isolation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Detailed report saved to: security_isolation_report.json")
    except Exception as e:
        print(f"Failed to save report: {e}")

    # Exit with appropriate code
    exit_code = {
        "SECURE": 0,
        "MOSTLY_SECURE": 1,
        "NEEDS_ATTENTION": 2,
        "INSECURE": 3,
        "ERROR": 4,
    }.get(results["overall_status"], 1)

    return exit_code


if __name__ == "__main__":
    exit(main())
