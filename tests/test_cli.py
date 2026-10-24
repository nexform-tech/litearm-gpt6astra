import json

from litearm_gpt6astra.cli import main


def test_schema_command(capsys):
    assert main(["schema"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["title"] == "litearm_astra_action"


def test_run_scripted(tmp_path):
    rc = main(
        [
            "run",
            "--mode",
            "scripted",
            "--seed",
            "0",
            "--out",
            str(tmp_path),
            "--no-viewer",
        ]
    )
    assert rc == 0
    assert (tmp_path / "result.json").is_file()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["success"] is True


def test_live_without_key_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("LITEARM_ASTRA_API_KEY", raising=False)
    rc = main(["run", "--mode", "live", "--out", str(tmp_path)])
    assert rc == 2
