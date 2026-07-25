"""
Tests for the HTTP multipart path of POST /api/v1/residues/upload.

The rest of the residue suite calls ResidueManager.store_residue_file()
directly, so nothing exercised the actual form-parsing chain
(FastAPI -> starlette.formparsers -> python_multipart). That chain is
dependency code we pin, and a residue file is the one place where a
byte-exact upload matters: the stored file is handed to a stage 2 worker
and its checksum is the authoritative identity for completion.

These tests are the regression guard for python-multipart upgrades.
"""
import pytest

from conftest import create_composite, get_test_engine

from app.main import app
from app.dependencies import get_residue_manager
from app.models.residues import ECMResidue


COMPOSITE_INT = 1000073001431003663
CHKCONST = 4294967291  # GMP-ECM's CHKSUMMOD, from ecm/ecm-ecm.h
UPLOAD_URL = "/api/v1/residues/upload"


def residue_line(b1: int = 50000, sigma: int = 12345,
                 param: int = 3, x: int = 0x1ABCD) -> str:
    """One valid GMP-ECM residue line with a correct checksum."""
    chk = b1
    chk *= sigma % CHKCONST
    chk *= COMPOSITE_INT % CHKCONST
    chk *= x % CHKCONST
    chk *= param + 1
    chk %= CHKCONST
    return (f"METHOD=ECM; B1={b1}; N={COMPOSITE_INT}; "
            f"SIGMA={sigma}; PARAM={param}; X={hex(x)}; CHECKSUM={chk};\n")


def residue_file(curves: int = 1, b1: int = 50000) -> bytes:
    """A residue file with `curves` distinct curve lines."""
    return "".join(
        residue_line(b1=b1, sigma=12345 + i, x=0x1ABCD + i * 0x1111)
        for i in range(curves)
    ).encode()


@pytest.fixture
def upload_client(client, tmp_path):
    """TestClient whose ResidueManager writes into a temp directory."""
    from app.services.residue_manager import ResidueManager

    manager = ResidueManager()
    manager.storage_dir = tmp_path
    app.dependency_overrides[get_residue_manager] = lambda: manager
    try:
        yield client, tmp_path
    finally:
        app.dependency_overrides.pop(get_residue_manager, None)


class TestResidueUploadMultipart:
    """The upload endpoint over real multipart/form-data."""

    def test_upload_parses_metadata_from_form_file(self, upload_client):
        """A valid multipart upload is accepted and its metadata parsed."""
        client, _ = upload_client
        create_composite(str(COMPOSITE_INT))

        response = client.post(
            UPLOAD_URL,
            files={"file": ("residues.txt", residue_file(), "text/plain")},
            headers={"X-Client-ID": "gpu-producer"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["composite"] == str(COMPOSITE_INT)
        assert body["b1"] == 50000
        assert body["parametrization"] == 3
        assert body["curve_count"] == 1

    def test_uploaded_bytes_are_stored_verbatim(self, upload_client):
        """Multipart decoding must not alter a single byte of the residue.

        A stage 2 worker re-reads this file and GMP-ECM verifies each
        line's checksum, so any mangling (line-ending translation,
        truncation, boundary bleed) corrupts the work silently.
        """
        client, _ = upload_client
        create_composite(str(COMPOSITE_INT))
        content = residue_file(curves=64)

        response = client.post(
            UPLOAD_URL,
            files={"file": ("residues.txt", content, "text/plain")},
            headers={"X-Client-ID": "gpu-producer"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["curve_count"] == 64
        assert body["file_size_bytes"] == len(content)

        _, TestingSessionLocal = get_test_engine()
        db = TestingSessionLocal()
        try:
            residue = db.query(ECMResidue).filter(
                ECMResidue.id == body["residue_id"]
            ).one()
            stored = open(residue.storage_path, "rb").read()
        finally:
            db.close()

        assert stored == content

    def test_missing_file_part_is_rejected(self, upload_client):
        """A request with no file part fails validation, not with a 500."""
        client, _ = upload_client
        create_composite(str(COMPOSITE_INT))

        response = client.post(
            UPLOAD_URL,
            headers={"X-Client-ID": "gpu-producer"},
        )

        assert response.status_code == 422
