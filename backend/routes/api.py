"""Canonical FastAPI routes for the NFL prediction API.

This module is intentionally declarative: each URL is registered here, while
the callable endpoint implementations live in ``backend.services.api_runtime``.
That keeps route discovery simple and prevents business logic from spreading
across route files.
"""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter

from backend.schemas_pipeline_status import (
    DatasetQualityStatus,
    ModelBundleStatus,
    PipelineStatusResponse,
)
from backend.services import api_runtime as svc

router = APIRouter()


def _route(
    path: str,
    endpoint: Any,
    methods: List[str],
    *,
    response_model: Any = None,
) -> None:
    """Register one path while preserving the service function signature."""
    router.add_api_route(
        path,
        endpoint=endpoint,
        methods=methods,
        response_model=response_model,
    )


_route("/health", svc.get_health_status, ["GET"], response_model=svc.HealthResponse)
_route("/status", svc.get_status_summary, ["GET"], response_model=svc.StatusResponse)
_route("/debug", svc.get_debug_snapshot, ["GET"])

_route(
    "/api/debug/predict-input",
    svc.inspect_prediction_input,
    ["POST"],
    response_model=svc.DebugPredictInputResponse,
)
_route(
    "/debug/predict-input",
    svc.inspect_prediction_input,
    ["POST"],
    response_model=svc.DebugPredictInputResponse,
)
_route(
    "/api/debug/dataset",
    svc.preview_dataset_rows,
    ["GET"],
    response_model=svc.DatasetPreviewResponse,
)
_route(
    "/debug/dataset",
    svc.preview_dataset_rows,
    ["GET"],
    response_model=svc.DatasetPreviewResponse,
)

_route("/status/overview", svc.get_status_overview, ["GET"], response_model=svc.StatusOverviewResponse)
_route("/health/pipeline", svc.get_pipeline_health, ["GET"], response_model=PipelineStatusResponse)
_route("/metadata/dataset", svc.get_dataset_metadata, ["GET"], response_model=DatasetQualityStatus)
_route("/metadata/model-bundle", svc.get_model_bundle_metadata, ["GET"], response_model=ModelBundleStatus)
_route("/metadata/model-learning", svc.get_model_learning_metadata, ["GET"])
_route("/api/metadata/model-learning", svc.get_model_learning_metadata, ["GET"])
_route("/artifacts/models/training-metrics-plot.png", svc.get_training_metrics_plot, ["GET"])
_route("/api/artifacts/models/training-metrics-plot.png", svc.get_training_metrics_plot, ["GET"])
_route(
    "/history/summary/memory",
    svc.get_memory_history_summary,
    ["GET"],
    response_model=svc.HistoryMetricsResponse,
)
_route("/status/models", svc.get_model_status, ["GET"])
_route("/status/runtime", svc.get_runtime_status, ["GET"], response_model=svc.RuntimeStatusResponse)
_route("/status/dataset-versioning", svc.get_dataset_versioning_status, ["GET"])
_route(
    "/status/performance-drift",
    svc.get_performance_drift_status,
    ["GET"],
    response_model=svc.PerformanceDriftResponse,
)
_route("/api/offseason/status", svc.get_offseason_status, ["GET"], response_model=svc.OffseasonStatusResponse)
_route("/offseason/status", svc.get_offseason_status, ["GET"], response_model=svc.OffseasonStatusResponse)

_route("/admin/retrain", svc.start_retrain_job, ["POST"], response_model=svc.RetrainResponse)
_route("/admin/retrain/{job_id}", svc.get_retrain_job_status, ["GET"], response_model=svc.RetrainJobStatus)
_route("/admin/promote/{job_id}", svc.promote_retrain_job, ["POST"], response_model=svc.PromoteResponse)

_route("/schedule", svc.get_schedule_slice, ["GET"], response_model=List[svc.ScheduleGameResponse])
_route("/schedule/next-week", svc.get_next_week_schedule, ["GET"], response_model=List[svc.ScheduleGameResponse])
_route("/api/predict/next-week", svc.get_next_week_prediction_inputs, ["GET"])
_route("/predict/next-week", svc.get_next_week_prediction_inputs, ["GET"])
_route("/api/teams/logos", svc.get_team_logo_metadata, ["GET"], response_model=svc.TeamLogosResponse)
_route("/teams/logos", svc.get_team_logo_metadata, ["GET"], response_model=svc.TeamLogosResponse)

_route("/history", svc.list_prediction_history, ["GET"], response_model=List[svc.HistoryEntryResponse])
_route("/history/summary", svc.get_prediction_history_summary, ["GET"], response_model=svc.HistorySummaryResponse)
_route("/history", svc.clear_prediction_history_for_request, ["DELETE"])

_route("/api/premium/explain", svc.explain_prediction_with_ai, ["POST"])
_route("/premium/explain", svc.explain_prediction_with_ai, ["POST"])
_route("/api/premium/chat", svc.chat_with_premium_ai, ["POST"])
_route("/premium/chat", svc.chat_with_premium_ai, ["POST"])

_route("/api/predict", svc.predict_game, ["POST"], response_model=svc.PredictionResponse)
_route("/predict", svc.predict_game, ["POST"], response_model=svc.PredictionResponse)
