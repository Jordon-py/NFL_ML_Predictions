# ==========================================
# File: backend/schemas.py
# Role: Pydantic schemas for API request/response validation.
# Input Data: Dict payloads and field values.
# Output Data: Validated model instances.
# Dependencies: __future__, typing, pydantic
# Notes: Single source of API shapes.
# ==========================================

"""
FILE: backend/schemas.py
PURPOSE: Canonical Pydantic models for NFL Prediction API contracts.
"""
from __future__ import annotations
import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int

class ScorePrediction(BaseModel):
    home_score: float
    away_score: float

class WinnerPrediction(BaseModel):
    winner: str
    proba_home: float = Field(..., ge=0.0, le=1.0)
    proba_away: float = Field(..., ge=0.0, le=1.0)
    proba_draw: Optional[float] = None

class SimulationMetrics(BaseModel):
    sim_home_score: float
    sim_away_score: float
    sim_home_sd: float
    sim_away_sd: float
    sim_n: int

class PredictionResponse(BaseModel):
    scores: ScorePrediction
    winner: WinnerPrediction
    simulation_metrics: Optional[SimulationMetrics] = None
    prediction_source: str
    win_classifier_used: bool = False

class UnifiedPredictionResponse(BaseModel):
    home_score: float
    away_score: float
    point_diff: float
    home_win_probability: float
    away_win_probability: float
    prediction_source: str
    win_classifier_used: bool
    simulation_metrics: Optional[Dict[str, Any]] = None
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    home_name: Optional[str] = None
    away_name: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    mode: str
    reason: str

class HistoryEntry(UnifiedPredictionResponse):
    ts: str
    user_id: Optional[str] = None
    storage_key: Optional[str] = None
    final_home_score: Optional[int] = None
    final_away_score: Optional[int] = None
    game_status: Optional[str] = None
    score_updated_at: Optional[str] = None

class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    total: int
    user_id: Optional[str] = None

class DatasetInfo(BaseModel):
    rows: int
    features: int

class HistoryMetrics(BaseModel):
    total_predictions: int
    win_rate: Optional[float] = None
    note: Optional[str] = None

class StatusOverviewResponse(BaseModel):
    health: HealthResponse
    dataset: DatasetInfo
    history: HistoryMetrics

class ScoreEntry(BaseModel):
    game_id: str
    season: Optional[int] = None
    week: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: Optional[str] = None
    updated_at: Optional[datetime.datetime] = None

class ScheduleEntry(BaseModel):
    home_team: str
    away_team: str
    season: int
    week: int
    kickoff: Optional[datetime.datetime] = None
    game_id: Optional[str] = None
    home_abbr: Optional[str] = None
    away_abbr: Optional[str] = None
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    stadium: Optional[str] = None

class ScheduleResponse(BaseModel):
    games: List[ScheduleEntry]

class TeamMeta(BaseModel):
    logoUrl: str
    name: Optional[str] = None
    primaryColor: Optional[str] = None
    secondaryColor: Optional[str] = None
    wordmark: Optional[str] = None

class TeamLogosResponse(BaseModel):
    teams: Dict[str, TeamMeta]

class SeasonContextResponse(BaseModel):
    phase: str
    label: str
    message: str
    current_season: int
    display_week: Optional[int] = None
    games_in_next_window: int
    next_kickoff: Optional[datetime.datetime] = None
    generated_at: datetime.datetime

class StoredPredictionRecord(HistoryEntry):
    request: PredictionRequest

class ExplainPredictionRequest(BaseModel):
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    season: Optional[int] = None
    week: Optional[int] = None
    prediction: Optional[Dict[str, Any]] = None

class ExplainPredictionResponse(BaseModel):
    game_id: Optional[str] = None
    used_llm: bool = False
    llm_model: Optional[str] = None
    explanation: str = ""
    bullets: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    prediction: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    used_llm: bool = False
    llm_model: Optional[str] = None
    error: Optional[str] = None

class AdminRetrainRequest(BaseModel):
    dataset_path: Optional[str] = None
    out_dir: Optional[str] = None
    force: bool = False

__all__ = [
    "PredictionRequest",
    "ScorePrediction",
    "WinnerPrediction",
    "SimulationMetrics",
    "PredictionResponse",
    "UnifiedPredictionResponse",
    "HealthResponse",
    "HistoryEntry",
    "HistoryResponse",
    "StoredPredictionRecord",
    "DatasetInfo",
    "HistoryMetrics",
    "StatusOverviewResponse",
    "ScheduleEntry",
    "ScheduleResponse",
    "TeamMeta",
    "TeamLogosResponse",
    "SeasonContextResponse",
    "ExplainPredictionRequest",
    "ExplainPredictionResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "AdminRetrainRequest",
    "ScoreEntry",
]
