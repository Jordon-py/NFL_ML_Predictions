import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import HistoryChart from "./HistoryChart.jsx";

describe("HistoryChart", () => {
  it("renders summary metrics and filters resolved predictions", () => {
    render(
      <HistoryChart
        summary={{
          total_predictions: 2,
          resolved_games: 1,
          win_rate: 1,
          avg_abs_spread_error: 2.5,
          avg_confidence: 0.68,
        }}
        history={[
          {
            ts: "2026-03-20T00:00:00.000Z",
            season: 2025,
            week: 1,
            home_team: "BUF",
            away_team: "KC",
            home_score: 24,
            away_score: 20,
            home_win_probability: 0.68,
            away_win_probability: 0.32,
            final_home_score: 27,
            final_away_score: 20,
          },
          {
            ts: "2026-03-22T00:00:00.000Z",
            season: 2025,
            week: 2,
            home_team: "PHI",
            away_team: "DAL",
            home_score: 21,
            away_score: 24,
            home_win_probability: 0.41,
            away_win_probability: 0.59,
          },
        ]}
      />
    );

    expect(screen.getAllByText("Resolved").length).toBeGreaterThan(0);
    expect(screen.getByText("100%")).toBeTruthy();
    expect(screen.getByText("2.5 pts")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Resolved" }));

    expect(screen.getByText(/KC@BUF/)).toBeTruthy();
    expect(screen.queryByText(/PHI@DAL/)).toBeNull();
  });
});
