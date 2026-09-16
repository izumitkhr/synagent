"""Locate the SEM MaiML / bitmap linked to an XRD MaiML via SampleDelivery UUID relay.

Linkage model:
  Each transfer mints a new material UUID for the sample, and every
  SampleDelivery MaiML (document name 'SampleDeliveryTTT') records the
  before/after pair. Union-Find over shared material UUIDs therefore
  connects the whole sequence transitively, e.g.:

    SP1_Log(45b0) - SP1toSEM(45b0,3897) - SEMtoSEM(3897,3f83)
      - sem_*.maiml(3f83) - SEMtoXRD(3f83,04c6) - XRD(04c6)

  The SEM MaiML (document name 'semDataFile') references its bitmap with a
  relative <uri> (./sem_*.bmp) resolved against its own directory.

All failures raise LinkageError: by design this pipeline stops on any
ambiguity instead of falling back (stale rotation, missing bitmap, ...).
"""

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

NS = {"maiml": "http://www.maiml.org/schemas"}
SEM_DOC_NAME = "semDataFile"


class LinkageError(RuntimeError):
    """Raised when the XRD->SEM linkage cannot be resolved unambiguously."""


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _parse_maiml(path):
    """Return (document_name, material_uuids) for one MaiML file."""
    root = ET.parse(path).getroot()
    name_elem = root.find(".//maiml:document/maiml:name", NS)
    doc_name = name_elem.text if name_elem is not None else None
    uuids = []
    for mat in root.findall(".//maiml:material", NS):
        u = mat.find("maiml:uuid", NS)
        if u is not None and u.text:
            uuids.append(u.text)
    return doc_name, uuids


def _bmp_uri(sem_maiml_path):
    """Return the bitmap path referenced by <uri> inside a SEM MaiML."""
    root = ET.parse(sem_maiml_path).getroot()
    for uri in root.findall(".//maiml:uri", NS):
        if uri.text and uri.text.strip().lower().endswith(".bmp"):
            return Path(sem_maiml_path).parent / uri.text.strip()
    raise LinkageError(
        f"No .bmp <uri> found inside SEM MaiML: {sem_maiml_path}"
    )


def find_linked_sem(xrd_maiml_path, raw_dir):
    """Find the SEM MaiML + bitmap belonging to the same sample as the XRD MaiML.

    xrd_maiml_path: the XRD MaiML copied into run_dir.
    raw_dir: instrument-side folder holding the latest MaiML/bitmap files
             (incl. SampleDelivery).

    Returns (sem_maiml_path, bmp_path). Raises LinkageError on any failure.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise LinkageError(f"raw_data_dir does not exist or is not a directory: {raw_dir}")

    _, xrd_uuids = _parse_maiml(xrd_maiml_path)
    if not xrd_uuids:
        raise LinkageError(f"No material UUID found in XRD MaiML: {xrd_maiml_path}")

    raw_files = sorted(raw_dir.glob("*.maiml"))
    if not raw_files:
        raise LinkageError(f"No .maiml files found in raw_data_dir: {raw_dir}")

    uf = _UnionFind()
    file_info = {}  # path -> (doc_name, uuids)
    for p in raw_files:
        try:
            doc_name, uuids = _parse_maiml(p)
        except ET.ParseError as e:
            logger.warning("Skipping unparsable MaiML %s: %s", p.name, e)
            continue
        file_info[p] = (doc_name, uuids)
        for u in uuids[1:]:
            uf.union(uuids[0], u)

    # Connect the run_dir XRD copy into the graph via its UUIDs.
    for u in xrd_uuids[1:]:
        uf.union(xrd_uuids[0], u)
    xrd_root = uf.find(xrd_uuids[0])

    sem_candidates = [
        p for p, (doc_name, uuids) in file_info.items()
        if doc_name == SEM_DOC_NAME
        and any(uf.find(u) == xrd_root for u in uuids)
    ]
    if not sem_candidates:
        linked = [p.name for p, (_, uuids) in file_info.items()
                  if any(uf.find(u) == xrd_root for u in uuids)]
        raise LinkageError(
            f"No SEM MaiML (document name '{SEM_DOC_NAME}') linked to "
            f"{Path(xrd_maiml_path).name} in {raw_dir}. "
            f"Files in the same linkage group: {linked or 'none'}. "
            f"The SampleDelivery chain may be incomplete (rotation?)."
        )
    if len(sem_candidates) > 1:
        names = ", ".join(p.name for p in sem_candidates)
        raise LinkageError(
            f"Multiple SEM MaiML files linked to {Path(xrd_maiml_path).name}: "
            f"{names}. Cannot disambiguate."
        )

    sem_maiml = sem_candidates[0]
    bmp_path = _bmp_uri(sem_maiml)
    if not bmp_path.is_file():
        raise LinkageError(
            f"SEM bitmap referenced by {sem_maiml.name} not found: {bmp_path}"
        )
    logger.info("Linked SEM MaiML: %s (bitmap: %s)", sem_maiml.name, bmp_path.name)
    return sem_maiml, bmp_path


def fetch_sem_image(xrd_maiml_path, raw_dir, dest_dir):
    """Resolve the linked SEM files and copy them into dest_dir (run_dir/data).

    Copies the SEM MaiML and bitmap, converts the bitmap to PNG for
    vision-LLM input, and returns {'sem_maiml', 'sem_bmp', 'sem_png'} paths.
    """
    from PIL import Image

    sem_maiml, bmp_path = find_linked_sem(xrd_maiml_path, raw_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    sem_maiml_dst = dest_dir / sem_maiml.name
    bmp_dst = dest_dir / bmp_path.name
    shutil.copy2(sem_maiml, sem_maiml_dst)
    shutil.copy2(bmp_path, bmp_dst)

    png_dst = bmp_dst.with_suffix(".png")
    Image.open(bmp_dst).save(png_dst)
    logger.info("SEM image ready: %s", png_dst)
    return {"sem_maiml": sem_maiml_dst, "sem_bmp": bmp_dst, "sem_png": png_dst}
