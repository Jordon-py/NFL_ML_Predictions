import { z } from "zod";

export const MatchupInputSchema = z.object({
  home_team: z.string(),
  away_team: z.string(),
  season: z.number(),
  week: z.number(),
});

export type MatchupInput = z.infer<typeof MatchupInputSchema>;

export const MatchupCardSchema = z.object({
  summary: z.string().default(""),
  confidence: z.number().min(0).max(1).default(0.5),
});

export const PredictWithNewsCardSchema = z.object({
  prediction_source: z.literal("model_only"),
  llm_card: z.null(),
});

const DISABLED_ERROR =
  "LLM/Ollama features are disabled for this release. Use backend /api/predict and /api/predict/explain only.";

export async function streamMatchupExplanation() {
  throw new Error(DISABLED_ERROR);
}

export async function generateMatchupCard() {
  throw new Error(DISABLED_ERROR);
}

export async function predictWithNewsCard() {
  return { prediction_source: "model_only", llm_card: null };
}

export async function predictWithNewsPrediction() {
  return { prediction_source: "model_only", llm_card: null };
}
