"""Lightweight self-check for the WhiteML-Box repository.

Run from the repository root:
    python scripts/check_repository.py
"""
from __future__ import annotations

import pathlib
import py_compile
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "mlbox_app.py",
    "docs/USER_GUIDE.md",
    "docs/MAPPING_WORKFLOW.md",
    "docs/SOFTWAREX_METADATA.md",
    "data/Spectra_Nitrogen.xlsx",
    "data/WhiteML-Box.tif",
]


def check_required_files() -> list[str]:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    return missing


def check_python_syntax() -> str:
    py_compile.compile(str(ROOT / "mlbox_app.py"), doraise=True)
    return "ok"


def check_xlsx() -> str:
    path = ROOT / "data" / "Spectra_Nitrogen.xlsx"
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        dim = root.find(".//main:dimension", ns).attrib.get("ref")
        if dim != "A1:BVY67":
            return f"warning: unexpected xlsx dimension {dim}"
    return "ok: Sheet1 dimension A1:BVY67"


def check_raster() -> str:
    try:
        import rasterio  # type: ignore
    except Exception:
        return "skipped: rasterio is not installed in this environment"
    with rasterio.open(ROOT / "data" / "WhiteML-Box.tif") as src:
        if src.count != 12:
            return f"warning: expected 12 bands, found {src.count}"
        return f"ok: {src.width}x{src.height}, {src.count} bands, CRS={src.crs}"


def main() -> int:
    missing = check_required_files()
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  - {p}")
        return 1

    print("Required files: ok")
    print(f"Python syntax: {check_python_syntax()}")
    print(f"Spreadsheet: {check_xlsx()}")
    print(f"Raster: {check_raster()}")
    print("Repository self-check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
