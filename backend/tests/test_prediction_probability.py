from backend.services import api_runtime as svc


def test_final_home_win_probability_blends_classifier_with_score_margin():
    close_away_edge = svc._finalize_home_win_probability(0.30, -1.5, clf_used=True)
    large_away_edge = svc._finalize_home_win_probability(0.30, -7.0, clf_used=True)
    home_score_edge = svc._finalize_home_win_probability(0.70, 7.0, clf_used=True)

    assert large_away_edge < close_away_edge
    assert close_away_edge < 0.50
    assert home_score_edge > 0.70


def test_final_home_win_probability_uses_score_margin_without_classifier():
    home_edge = svc._finalize_home_win_probability(0.50, 3.0, clf_used=False)
    away_edge = svc._finalize_home_win_probability(0.50, -3.0, clf_used=False)

    assert home_edge > 0.50
    assert away_edge < 0.50
