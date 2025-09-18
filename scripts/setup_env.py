#scripts/setup_env.py


import shutil
import sys
import subprocess


def check_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_tesseract():
    if not check_cmd("tesseract"):
        print("[!] Tesseract not found. Install from: https://github.com/tesseract-ocr/tesseract")
        return False
    return True


def check_ghostscript():
    if not (check_cmd("gs") or check_cmd("gswin64c")):
        print("[!] Ghostscript not found. Install from: https://ghostscript.com/releases/gsdnld.html")
        return False
    return True


def check_poppler():
    if not check_cmd("pdftoppm"):
        print("[!] Poppler not found. On Windows, install from: http://blog.alivate.com.au/poppler-windows/")
        return False
    return True


def main():
    print("=== Checking environment dependencies ===")
    ok = True
    if not check_tesseract():
        ok = False
    if not check_ghostscript():
        ok = False
    if not check_poppler():
        ok = False

    if ok:
        print("[+] All required system dependencies are available.")
    else:
        print("[!] Missing dependencies detected. Please install them before running the app.")


if __name__ == "__main__":
    main()