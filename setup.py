#!/usr/bin/env python3
"""
Setup & launcher for Face Attendance System
Run this first: python setup.py
"""
import subprocess, sys, os

print("=" * 55)
print("  Face Attendance System — Setup")
print("=" * 55)

packages = [
    "flask",
    "opencv-python",
    "opencv-contrib-python",
    "numpy",
    "pandas",
]

print("\nInstalling dependencies...")
for pkg in packages:
    print(f"  • {pkg}", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "-q"],
        capture_output=True
    )
    print(" ✅" if result.returncode == 0 else f" ❌ ({result.stderr.decode()[:80]})")

print("\nChecking OpenCV face module...")
try:
    import cv2
    r = cv2.face.LBPHFaceRecognizer_create()
    print("  • LBPHFaceRecognizer ✅")
except Exception as e:
    print(f"  • cv2.face not found — installing opencv-contrib-python")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "opencv-contrib-python", "--break-system-packages", "-q"])

print("\nSetup complete!")
print("\nTo start the server, run:")
print("  python app.py")
print("\nThen open: http://localhost:5050")
print("=" * 55)
