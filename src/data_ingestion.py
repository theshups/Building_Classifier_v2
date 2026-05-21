"""
src/data_ingestion.py — BuildingYOLO
======================================
Auto-downloads all four class datasets. No manual upload.

Sources:
  exterior_facade  CMP Facade DB   55 MB   http://cmp.felk.cvut.cz
  office_interior  MIT Indoor      2.4 GB  http://groups.csail.mit.edu
  warehouse        MIT Indoor      same archive
  pipelines        Roboflow API    utp-jtbn5/pipeline-tracks

Roboflow fix:
  SDK downloads into a versioned subfolder e.g. pipeline-tracks-1/
  We use rglob to find all images at any depth, then match sibling labels.

Requires:
  ROBOFLOW_API_KEY env var — run setup_roboflow.bat once.
"""

import os
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger    import get_logger
from exception import AppException

log = get_logger(__name__)

# ── SSL bypass for Windows ────────────────────────────────────────────────
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE
ssl._create_default_https_context = ssl._create_unverified_context

# ── Dataset URLs ──────────────────────────────────────────────────────────
FACADE_URL = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
MIT_URL    = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"

DATA_ROOT = Path("data/raw")
MAX_IMG   = 1200   # cap per class

# MIT Indoor categories → our class labels
MIT_MAP = {
    "office":          "office_interior",
    "meeting_room":    "office_interior",
    "conference_room": "office_interior",
    "computer_room":   "office_interior",
    "warehouse":       "warehouse",
    "garage":          "warehouse",
    "storage_room":    "warehouse",
    "industrial":      "warehouse",
}
EXTRACT_ONLY = set(MIT_MAP.keys())

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class DataIngestion:
    def __init__(self, skip_mit: bool = False):
        """
        skip_mit : set True to skip the 2.4 GB MIT download
                   (office_interior and warehouse will have 0 images).
        """
        self.skip_mit = skip_mit
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────
    def run(self) -> dict:
        try:
            log.info("=" * 60)
            log.info("  BuildingYOLO  -  Data Ingestion")
            log.info("  exterior_facade : CMP Facade DB  (~55 MB)")
            log.info("  office_interior : MIT Indoor     (~2.4 GB)")
            log.info("  warehouse       : MIT Indoor     (same)")
            log.info("  pipelines       : Roboflow API")
            log.info("=" * 60)

            images: dict = {}

            # 1. CMP Facade
            images["exterior_facade"] = self._get_cmp_facade()

            # 2. MIT Indoor (office + warehouse)
            if self.skip_mit:
                log.warning("--skip-mit: office_interior and warehouse skipped.")
                images["office_interior"] = []
                images["warehouse"]       = []
            else:
                mit = self._get_mit_indoor()
                images["office_interior"] = mit["office_interior"]
                images["warehouse"]       = mit["warehouse"]

            # 3. Roboflow pipelines
            rf = self._get_roboflow_pipelines()
            images["pipelines"]       = rf["images"]
            images["pipeline_labels"] = rf["labels"]

            # Summary
            log.info("-" * 60)
            for cls, v in images.items():
                if cls != "pipeline_labels":
                    log.info(f"  {cls:<20} {len(v):>5} images")
            log.info("=" * 60)

            return images

        except Exception as exc:
            raise AppException(exc, sys) from exc

    # ── 1. CMP Facade DB ─────────────────────────────────────────────────
    def _get_cmp_facade(self) -> list:
        fdir = DATA_ROOT / "cmp_facade"

        if fdir.exists():
            imgs = self._collect_images(fdir)
            if imgs:
                log.info(f"CMP Facade: already on disk ({len(imgs)} images).")
                return imgs

        z = DATA_ROOT / "cmp_facade.zip"
        if not z.exists():
            self._http_download(FACADE_URL, z, "CMP Facade DB (~55 MB)")

        log.info("Extracting CMP Facade ...")
        fdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(fdir)
        z.unlink(missing_ok=True)

        imgs = self._collect_images(fdir)
        log.info(f"exterior_facade: {len(imgs)} images")
        return imgs

    # ── 2. MIT Indoor Scenes ──────────────────────────────────────────────
    def _get_mit_indoor(self) -> dict:
        mdir   = DATA_ROOT / "mit_indoor"
        result = {"office_interior": [], "warehouse": []}

        if mdir.exists() and any(mdir.rglob("*.jpg")):
            log.info("MIT Indoor: already on disk. Scanning ...")
            for root, _, files in os.walk(str(mdir)):
                folder = Path(root).name.lower()
                if folder in MIT_MAP:
                    label = MIT_MAP[folder]
                    imgs  = [Path(root) / f for f in files
                             if f.lower().endswith(IMG_EXTS)]
                    if imgs:
                        result[label].extend(imgs)
            log.info(f"office_interior: {len(result['office_interior'])} images")
            log.info(f"warehouse      : {len(result['warehouse'])} images")
            return result

        tar = DATA_ROOT / "indoorCVPR_09.tar"
        if not tar.exists():
            self._http_download(MIT_URL, tar, "MIT Indoor Scenes (~2.4 GB)")

        log.info("Extracting MIT Indoor (selected categories only) ...")
        mdir.mkdir(parents=True, exist_ok=True)
        self._selective_tar(tar, mdir)
        tar.unlink(missing_ok=True)

        for root, _, files in os.walk(str(mdir)):
            folder = Path(root).name.lower()
            if folder in MIT_MAP:
                label = MIT_MAP[folder]
                imgs  = [Path(root) / f for f in files
                         if f.lower().endswith(IMG_EXTS)]
                if imgs:
                    result[label].extend(imgs)
                    log.info(f"  MIT '{folder}' -> '{label}': {len(imgs)}")

        log.info(f"office_interior: {len(result['office_interior'])} total")
        log.info(f"warehouse      : {len(result['warehouse'])} total")
        return result

    # ── 3. Roboflow pipeline-tracks ───────────────────────────────────────
    def _get_roboflow_pipelines(self) -> dict:
        pipe_dir = DATA_ROOT / "pipeline_images"
        lbl_dir  = DATA_ROOT / "pipeline_labels"

        # Return cached if already downloaded
        if pipe_dir.exists() and lbl_dir.exists():
            existing = self._collect_images(pipe_dir)
            if len(existing) >= 80:
                labels = {
                    p.stem: lbl_dir / (p.stem + ".txt")
                    for p in existing
                    if (lbl_dir / (p.stem + ".txt")).exists()
                }
                log.info(f"Pipelines: already on disk "
                         f"({len(existing)} images, {len(labels)} labels).")
                return {"images": existing, "labels": labels}

        # Validate API key
        key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        if not key:
            log.error("=" * 60)
            log.error("  ROBOFLOW_API_KEY is not set!")
            log.error("  Run setup_roboflow.bat, then reopen PowerShell.")
            log.error("=" * 60)
            raise AppException("ROBOFLOW_API_KEY missing. Run setup_roboflow.bat", sys)

        return self._roboflow_download(pipe_dir, lbl_dir, key)

    def _roboflow_download(self, pipe_dir: Path,
                            lbl_dir: Path, key: str) -> dict:
        try:
            from roboflow import Roboflow
        except ImportError:
            raise AppException(
                "roboflow package not installed. Run install.bat", sys)

        pipe_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True,  exist_ok=True)
        raw = DATA_ROOT / "roboflow_raw"
        raw.mkdir(parents=True, exist_ok=True)

        # ── Connect and download ──────────────────────────────────────────
        try:
            log.info("Connecting to Roboflow ...")
            rf = Roboflow(api_key=key)

            log.info("Loading workspace: utp-jtbn5 ...")
            ws = rf.workspace("utp-jtbn5")

            log.info("Loading project: pipeline-tracks ...")
            project = ws.project("pipeline-tracks")

            # Try version 1 first; version() returns available versions
            log.info("Downloading YOLOv8 format dataset ...")
            try:
                dataset = project.version(1).download(
                    "yolov8", location=str(raw), overwrite=True
                )
            except Exception as ver_err:
                log.warning(f"Version 1 failed ({ver_err}). Trying version 2 ...")
                dataset = project.version(2).download(
                    "yolov8", location=str(raw), overwrite=True
                )

            log.info("Roboflow download complete.")
            if dataset:
                log.info(f"  Dataset location: {getattr(dataset, 'location', raw)}")

        except AppException:
            raise
        except Exception as exc:
            raise AppException(f"Roboflow connection failed: {exc}", sys) from exc

        # ── Find all images at any depth ──────────────────────────────────
        log.info(f"Scanning downloaded files in {raw} ...")
        self._debug_folder(raw)   # log structure for debugging

        all_imgs = [
            p for p in raw.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and p.is_file()
            and "label" not in p.parent.name.lower()  # skip labels/ folders
        ]

        if not all_imgs:
            log.error("No images found after Roboflow download!")
            log.error("Raw folder contents listed above.")
            raise AppException(
                "Roboflow downloaded 0 images. "
                "Check your API key and dataset version.", sys)

        log.info(f"Found {len(all_imgs)} pipeline images.")

        # ── Copy images + labels ──────────────────────────────────────────
        images: list = []
        labels: dict = {}

        for img_path in all_imgs:
            dst_img = pipe_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            images.append(dst_img)

            # Label is in sibling labels/ folder at same level as images/
            lbl_src = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if not lbl_src.exists():
                # Also try same folder
                lbl_src = img_path.parent / (img_path.stem + ".txt")
            if not lbl_src.exists():
                # Search recursively near the image
                candidates = list(raw.rglob(img_path.stem + ".txt"))
                lbl_src = candidates[0] if candidates else lbl_src

            if lbl_src.exists():
                dst_lbl = lbl_dir / (img_path.stem + ".txt")
                if not dst_lbl.exists():
                    shutil.copy2(lbl_src, dst_lbl)
                labels[img_path.stem] = dst_lbl

        log.info(f"Pipelines: {len(images)} images, {len(labels)} labels")
        return {"images": images, "labels": labels}

    def _debug_folder(self, root: Path, max_lines: int = 40):
        """Log folder tree so we can see what Roboflow actually downloaded."""
        log.info(f"  Folder tree ({root}):")
        count = 0
        for p in sorted(root.rglob("*")):
            if count >= max_lines:
                log.info("  ... (truncated)")
                break
            indent = "  " + "  " * (len(p.relative_to(root).parts) - 1)
            log.info(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
            count += 1

    # ── Helpers ───────────────────────────────────────────────────────────
    def _collect_images(self, directory: Path) -> list:
        imgs = []
        for ext in IMG_EXTS:
            imgs.extend(directory.rglob(f"*{ext}"))
        return imgs

    def _http_download(self, url: str, dest: Path, label: str):
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading: {label}")
        log.info(f"  URL: {url}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "BuildingYOLO/1.0"})
        try:
            with urllib.request.urlopen(req, context=_SSL, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0))
                done  = 0
                last_pct = -1
                with open(dest, "wb") as f:
                    while True:
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = int(done / total * 100)
                            if pct >= last_pct + 10:
                                log.info(f"  {pct}%  "
                                         f"({done >> 20}/{total >> 20} MB)")
                                last_pct = pct
        except Exception as exc:
            dest.unlink(missing_ok=True)
            raise AppException(f"Download failed [{url}]: {exc}", sys) from exc
        log.info(f"  Saved -> {dest}  ({dest.stat().st_size >> 20} MB)")

    def _selective_tar(self, tar_path: Path, dest: Path):
        """Extract only the MIT Indoor categories we need."""
        with tarfile.open(tar_path, "r:*") as tar:
            all_members = tar.getmembers()
            keep = [
                m for m in all_members
                if len(Path(m.name).parts) >= 3
                and Path(m.name).parts[-2].lower() in EXTRACT_ONLY
            ]
            log.info(f"  {len(keep)}/{len(all_members)} files selected.")
            for i, m in enumerate(keep):
                try:
                    tar.extract(m, dest)
                except Exception as exc:
                    log.warning(f"  Skipped {m.name}: {exc}")
                if i % 500 == 0 and i:
                    log.info(f"  Extracted {i}/{len(keep)} ...")
        log.info("  Extraction complete.")
