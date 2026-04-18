from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import github_token_warning_for_dimensions
from grading_utils import RepoSpec, make_dimension_comments, scan_api_serving, scan_release_discipline


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_release_after_cutoff_does_not_count(monkeypatch):
    payload = [
        {
            "tag_name": "v2",
            "target_commitish": "main",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-03-28T00:10:00Z",
        },
        {
            "tag_name": "v1",
            "target_commitish": "main",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-03-27T20:00:00Z",
        },
    ]
    monkeypatch.setattr("grading_utils.requests.get", lambda *args, **kwargs: _FakeResponse(payload))

    evidence = scan_release_discipline(
        RepoSpec("repo", "https://github.com/org/repo"),
        cutoff_str="2026-03-27 23:59:59",
        timezone_name="Europe/Madrid",
    )

    assert evidence["release_found"] == 1
    assert evidence["release_tag_name"] == "v1"
    assert evidence["release_published_at_used"] == "2026-03-27T20:00:00Z"
    assert evidence["release_count_found"] == 1


def test_api_serving_rejects_inline_model_predict(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "api.py").write_text(
        """
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = object()

class PredictRequest(BaseModel):
    x: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest):
    return {"prediction": model.predict([[payload.x]])[0]}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    evidence = scan_api_serving(tmp_path)

    assert evidence["api_fastapi_app_present"] == 1
    assert evidence["api_predict_endpoint_present"] == 1
    assert evidence["api_predict_calls_inference_logic"] == 0
    assert evidence["api_serving_cap_reason"] == "predict_without_inference_logic"


def test_monitoring_comment_stays_documentation_honest():
    comments = make_dimension_comments(
        evidence={
            "monitoring_local_runtime_log_signal": 0,
            "monitoring_api_request_trace_signal": 0,
            "monitoring_api_logging_signal": 0,
            "monitoring_wandb_inference_telemetry_signal": 0,
            "monitoring_healthcheck_signal": 0,
            "monitoring_render_runtime_documented": 1,
        },
        scores={},
        selected_dimensions={"monitoring"},
    )

    assert "documented" in comments["monitoring_comment"].lower()
    assert "render" in comments["monitoring_comment"].lower()


def test_missing_github_token_warning_is_explicit(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    warning = github_token_warning_for_dimensions({"release_discipline", "deployment"})

    assert warning is not None
    assert "GITHUB_TOKEN is missing" in warning
    assert "release_discipline" in warning
