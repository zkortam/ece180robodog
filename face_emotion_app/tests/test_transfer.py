"""Portable enrollment bundles: enrol on any machine, run on the robot.

Only the compact prototypes travel -- a 128-float face embedding and averaged
expression vectors. No images. The file arrives by email or chat, so it is
validated rather than trusted: a malformed or hostile one must produce a clean
error and must never leave the database half-written.
"""
import threading

import numpy as np
import pytest

import face_emotion as fe


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def engine(tmp_path):
    e = object.__new__(fe.FaceEngine)
    e.db_path = tmp_path / "enrollments.json"
    e.emotion_db_path = tmp_path / "emotions.json"
    e.db = {}
    e.emotion_db = {}
    e._stamp_dbs()
    e.lock = threading.Lock()
    return e


# ----------------------------------------------------------------- round trip

def test_a_bundle_survives_export_and_import(tmp_path):
    src = engine(tmp_path / "a")
    src.db["alex"] = unit(1, 2, 3)
    src.emotion_db["alex"] = {"happy": np.arange(7, dtype=np.float32)}
    bundle = src.export_people()

    dst = engine(tmp_path / "b")
    result = dst.import_people(bundle)

    assert result["people"] == ["alex"]
    np.testing.assert_allclose(dst.db["alex"], src.db["alex"], atol=1e-6)
    np.testing.assert_allclose(dst.emotion_db["alex"]["happy"],
                               src.emotion_db["alex"]["happy"], atol=1e-6)


def test_the_bundle_carries_no_images(tmp_path):
    """The privacy promise: prototypes only."""
    src = engine(tmp_path / "a")
    src.db["alex"] = unit(1, 0, 0)
    entry = src.export_people()["people"]["alex"]
    assert set(entry) <= {"face", "expressions"}


def test_exporting_one_person_excludes_the_others(tmp_path):
    src = engine(tmp_path / "a")
    src.db.update({"alex": unit(1, 0, 0), "sam": unit(0, 1, 0)})
    assert sorted(src.export_people(["alex"])["people"]) == ["alex"]


def test_importing_merges_rather_than_replacing(tmp_path):
    """Loading a friend's file must not wipe everyone already on the robot."""
    dst = engine(tmp_path / "b")
    dst.db["existing"] = unit(1, 0, 0)
    dst.import_people({"version": 1, "people": {"alex": {"face": unit(0, 1, 0).tolist()}}})
    assert sorted(dst.db) == ["alex", "existing"]


def test_an_existing_person_can_be_protected(tmp_path):
    dst = engine(tmp_path / "b")
    dst.db["alex"] = unit(1, 0, 0)
    result = dst.import_people(
        {"version": 1, "people": {"alex": {"face": unit(0, 1, 0).tolist()}}}, overwrite=False)
    assert result["skipped"] == ["alex"]
    np.testing.assert_allclose(dst.db["alex"], unit(1, 0, 0), atol=1e-6)


def test_an_imported_embedding_is_renormalized(tmp_path):
    """Matching is a dot product; an un-normalized vector silently rescales every
    score it ever participates in."""
    dst = engine(tmp_path / "b")
    dst.import_people({"version": 1, "people": {"alex": {"face": [3.0, 4.0, 0.0]}}})
    assert float(np.linalg.norm(dst.db["alex"])) == pytest.approx(1.0, abs=1e-6)


def test_the_import_is_persisted(tmp_path):
    dst = engine(tmp_path / "b")
    dst.import_people({"version": 1, "people": {"alex": {"face": unit(1, 0, 0).tolist()}}})
    assert sorted(fe.load_db(dst.db_path)) == ["alex"]


# --------------------------------------------------------------- bad bundles

@pytest.mark.parametrize("bundle,reason", [
    ("not a dict", "object"),
    ({"people": {}}, "version"),
    ({"version": 99, "people": {}}, "version"),
    ({"version": 1}, "no people"),
    ({"version": 1, "people": {}}, "no people"),
    ({"version": 1, "people": {"": {"face": [1.0]}}}, "unnamed"),
    ({"version": 1, "people": {"a": "nonsense"}}, "malformed"),
    ({"version": 1, "people": {"a": {"face": []}}}, "empty"),
    ({"version": 1, "people": {"a": {"face": [0.0, 0.0]}}}, "zero magnitude"),
])
def test_a_malformed_bundle_is_rejected_cleanly(tmp_path, bundle, reason):
    with pytest.raises(ValueError):
        engine(tmp_path / "b").import_people(bundle)


def test_a_nan_embedding_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        engine(tmp_path / "b").import_people(
            {"version": 1, "people": {"a": {"face": [float("nan"), 1.0]}}})


def test_a_wrong_sized_embedding_is_rejected(tmp_path):
    """A 5-float vector next to 128-float ones would break every dot product."""
    dst = engine(tmp_path / "b")
    dst.db["existing"] = np.ones(128, dtype=np.float32) / np.sqrt(128)
    with pytest.raises(ValueError, match="expected 128"):
        dst.import_people({"version": 1, "people": {"a": {"face": [1.0] * 5}}})


def test_a_rejected_bundle_leaves_the_database_untouched(tmp_path):
    """Validation happens before ANY write, so a bad file cannot half-apply."""
    dst = engine(tmp_path / "b")
    dst.db["existing"] = unit(1, 0, 0)
    dst._save_dbs(identities=True)
    with pytest.raises(ValueError):
        dst.import_people({"version": 1, "people": {
            "good": {"face": unit(0, 1, 0).tolist()},
            "bad": {"face": [0.0, 0.0, 0.0]},
        }})
    assert sorted(fe.load_db(dst.db_path)) == ["existing"]


# ------------------------------------------------------------------ filename

@pytest.mark.parametrize("stem,expected", [
    ("alex", "alex"), ("../../etc/passwd", "etcpasswd"), ("", "enrollments"),
    ('a"b', "ab"), ("with space", "withspace"),
])
def test_download_filenames_cannot_smuggle_a_path_or_header_break(stem, expected):
    assert fe._safe_filename(stem) == expected
