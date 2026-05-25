import re

from backend.app.core.settings import Settings, VERCEL_PROJECT_ORIGIN_REGEX


def test_default_vercel_regex_matches_expected_project_origins():
    settings = Settings(
        _env_file=None,
        app_env="production",
        allowed_origins_raw="https://new-nfl-predict.vercel.app",
        allow_origin_regex=None,
        allow_vercel_previews=True,
        restrict_cors=True,
    )

    pattern = settings.effective_allow_origin_regex
    assert pattern == VERCEL_PROJECT_ORIGIN_REGEX

    allowed_origins = [
        "https://nfl-predict-jr8e6ydim-christopher-jordons-projects.vercel.app",
        "https://nfl-ml-predictions-9f1ilaozw-christopher-jordons-projects.vercel.app",
        "https://frontend-azure-psi-82.vercel.app",
        "https://nfl-ml-predictionsoff-2fvuj9ix4-christopher-jordons-projects.vercel.app",
        "https://nflmlforcast.vercel.app",
    ]

    for origin in allowed_origins:
        assert re.fullmatch(pattern, origin), origin


def test_default_vercel_regex_is_origin_only_and_rejects_non_matching_hosts():
    assert re.fullmatch(VERCEL_PROJECT_ORIGIN_REGEX, "https://nflmlforcast.vercel.app")
    assert not re.fullmatch(VERCEL_PROJECT_ORIGIN_REGEX, "https://nflmlforcast.vercel.app/app")
    assert not re.fullmatch(VERCEL_PROJECT_ORIGIN_REGEX, "http://nflmlforcast.vercel.app")
    assert not re.fullmatch(VERCEL_PROJECT_ORIGIN_REGEX, "https://example.com")

def test_invalid_or_legacy_regex_falls_back_to_vercel_pattern():
    settings = Settings(
        _env_file=None,
        app_env="production",
        allowed_origins_raw="https://new-nfl-predict.vercel.app",
        allow_origin_regex="https://.*/.vercel/.app$",
        allow_vercel_previews=True,
        restrict_cors=True,
    )

    assert settings.effective_allow_origin_regex == VERCEL_PROJECT_ORIGIN_REGEX
