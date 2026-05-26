import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import TeamGrid from "./TeamGrid.jsx";

const games = [
  {
    season: 2025,
    week: 1,
    away_team: "KC",
    away_abbr: "KC",
    home_team: "BUF",
    home_abbr: "BUF",
    stadium: "Highmark Stadium",
    kickoff: "2025-09-07T20:25:00Z",
  },
  {
    season: 2025,
    week: 1,
    away_team: "DAL",
    away_abbr: "DAL",
    home_team: "PHI",
    home_abbr: "PHI",
    stadium: "Lincoln Financial Field",
    kickoff: "2025-09-07T17:00:00Z",
  },
];

afterEach(() => {
  cleanup();
});

describe("TeamGrid slate filters", () => {
  it("filters by team search and predicts only the visible slate", () => {
    const onPredictAll = vi.fn();

    render(<TeamGrid week={1} games={games} onPredictAll={onPredictAll} />);

    fireEvent.change(screen.getByLabelText("Search games by team or stadium"), {
      target: { value: "DAL" },
    });

    expect(screen.queryByText("Buffalo Bills")).toBeNull();
    expect(screen.getByText("Philadelphia Eagles")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Predict visible" }));

    expect(onPredictAll).toHaveBeenCalledTimes(1);
    expect(onPredictAll.mock.calls[0][0]).toHaveLength(1);
    expect(onPredictAll.mock.calls[0][0][0].away_team).toBe("DAL");
  });

  it("filters by prediction status", () => {
    const predictions = {
      "2025-1-BUF-KC": {
        home_score: 27,
        away_score: 20,
        home_win_probability: 0.7,
        away_win_probability: 0.3,
      },
    };

    render(<TeamGrid week={1} games={games} predictions={predictions} />);

    fireEvent.change(screen.getByLabelText("Filter games by prediction state"), {
      target: { value: "open" },
    });

    expect(screen.queryByText("Buffalo Bills")).toBeNull();
    expect(screen.getByText("Philadelphia Eagles")).toBeTruthy();
  });
});
