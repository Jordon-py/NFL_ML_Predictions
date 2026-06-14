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

  it("normalizes Rams schedule aliases before prediction", () => {
    const onPredict = vi.fn();
    const ramsGame = {
      season: 2026,
      week: 1,
      away_team: "SF",
      away_abbr: "SF",
      home_team: "LA",
      home_abbr: "LA",
      home_name: "Los Angeles Rams",
      away_name: "San Francisco 49ers",
    };

    render(<TeamGrid week={1} games={[ramsGame]} onPredict={onPredict} />);

    expect(screen.getByText("LAR")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Generate prediction" }));

    expect(onPredict).toHaveBeenCalledTimes(1);
    expect(onPredict.mock.calls[0][0].home_team).toBe("LAR");
    expect(onPredict.mock.calls[0][0].home_abbr).toBe("LAR");
  });

  it("renders Gemma expert reasoning from the prediction response", async () => {
    const expertReasoning =
      "Kansas City's road efficiency keeps this close. Buffalo's home win profile and healthier defensive context support the edge. The calibrated model projection is strongest at 27-23 with a 64% Bills confidence.";
    const predictions = {
      "2025-1-BUF-KC": {
        game_id: "2025_01_KC_BUF",
        home_score: 27,
        away_score: 23,
        home_win_probability: 0.64,
        away_win_probability: 0.36,
        prediction_source: "gemma_cloud_expert_calibrated",
        expert_reasoning: expertReasoning,
        expert_prediction: {
          used_llm: true,
          reasoning: expertReasoning,
        },
      },
    };

    render(<TeamGrid week={1} games={games} predictions={predictions} />);

    expect(screen.getByText("Gemma Cloud Expert Layer")).toBeTruthy();
    expect(await screen.findByText(/Buffalo's home win profile/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /Hide Premium AI Breakdown/i }));
    expect(screen.queryByText(/Buffalo's home win profile/)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /View Premium AI Breakdown/i }));
    expect(await screen.findByText(/calibrated model projection/)).toBeTruthy();
  });
});
