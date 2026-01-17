// api/ai/aiPredictMatchup.ts
import { ollama, generateText, streamText } from "ai-sdk-ollama";
import { Output, tool, stepCountIs } from "ai";
import { z } from "zod";
import { fetchJson } from "./fetch";
const process = require("node:process");
// -----------------------------
// Env (server-only)
// -----------------------------
const FASTAPI_BASE_URL =
  process.env.FASTAPI_BASE_URL ||
  process.env.NFL_API_BASE_URL ||
  "http://localhost:8000";

const ENABLE_WEB_SEARCH = process.env.ENABLE_WEB_SEARCH === "true";
// Gate web search so we do not attempt cloud-only features without a key.
const HAS_OLLAMA_API_KEY = Boolean(process.env.OLLAMA_API_KEY);

// Local model (runs on your machine)
const MODEL_LOCAL = process.env.OLLAMA_MODEL_LOCAL || "llama3.2";

// Cloud model (recommended for web-search agents)
const MODEL_CLOUD = process.env.OLLAMA_MODEL_CLOUD || "qwen3-coder:480b-cloud";

// -----------------------------
// Schemas
// -----------------------------
export const MatchupInputSchema = z.object({
  season: z.number().int(),
  week: z.number().int(),
  home_team: z.string().min(2),
  away_team: z.string().min(2),
});
export type MatchupInput = z.infer<typeof MatchupInputSchema>;

export const MatchupCardSchema = z.object({
  headline: z.string().min(5).max(140),
  bullets: z.array(z.string().min(3).max(140)).min(2).max(5),
  predictedScore: z.object({
    home: z.number(),
    away: z.number(),
  }),
  confidence: z.enum(["low", "medium", "high"]),
});

export const PredictWithNewsCardSchema = z.object({
  headline: z.string(),
  winner: z.string(),
  confidence: z.number().min(0).max(1),
  predictedScore: z.string(),

  baseModelSignals: z.object({
    homeTeam: z.string(),
    awayTeam: z.string(),
    homeWinProb: z.number().min(0).max(1),
    pointDiff: z.number(),
    total: z.number(),
  }),

  news: z.object({
    summary: z.string(),
    keyInjuries: z.array(z.string()).max(8),
    sources: z
      .array(
        z.object({
          title: z.string(),
          url: z.string(),
        })
      )
      .max(6),
  }),

  rationaleBullets: z.array(z.string()).min(3).max(7),
  riskFactors: z.array(z.string()).min(1).max(5),

  finalPick: z.object({
    lean: z.enum(["HOME", "AWAY"]),
    adjustedHomeWinProb: z.number().min(0).max(1),
    whyThisPick: z.string(),
  }),
});

// -----------------------------
// Helpers
// -----------------------------
function apiUrl(path: string) {
  const base = FASTAPI_BASE_URL.endsWith("/")
    ? FASTAPI_BASE_URL
    : `${FASTAPI_BASE_URL}/`;
  return new URL(path.replace(/^\//, ""), base).toString();
}

function clamp01(n: number) {
  if (!Number.isFinite(n)) return 0.5;
  return Math.max(0, Math.min(1, n));
}

function safeParseJson(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

type PredictionOptions = {
  record?: boolean;
};

// -----------------------------
// Fetch baseline prediction from FastAPI
// -----------------------------
async function fetchPrediction(
  input: MatchupInput,
  options: PredictionOptions = {}
) {
  const record = options.record !== false;
  const path = record ? "/api/predict" : "/api/predict?record=0";

  // record=false avoids polluting history during bulk UI predictions.
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  const raw = await res.text();
  const data = safeParseJson(raw);

  if (!res.ok) {
    throw new Error(`FastAPI /api/predict failed: ${res.status} ${raw}`);
  }

  const scores = data?.scores || {};
  const winner = data?.winner || {};

  const homeScore = Number(scores.home_score ?? data?.predicted_home_score ?? 0);
  const awayScore = Number(scores.away_score ?? data?.predicted_away_score ?? 0);

  const homeWinProb = clamp01(
    Number(winner.proba_home ?? data?.home_win_probability ?? 0.5)
  );

  return {
    ...data,
    modelSignals: {
      homeTeam: String(input.home_team).toUpperCase(),
      awayTeam: String(input.away_team).toUpperCase(),
      homeWinProb,
      pointDiff: homeScore - awayScore,
      total: homeScore + awayScore,
    },
    predictedScore: { home: homeScore, away: awayScore },
  };
}

// -----------------------------
// Tool: FastAPI /api/predict
// -----------------------------
const getPrediction = tool({
  description: "Get ML baseline prediction for a matchup from FastAPI /api/predict",
  inputSchema: MatchupInputSchema,
  execute: async (input: MatchupInput) => fetchPrediction(input),
});

// -----------------------------
// Model config
// -----------------------------
function getModel({ cloud }: { cloud: boolean }) {
  const modelName = cloud ? MODEL_CLOUD : MODEL_LOCAL;

  // web-search agents can be very token-hungry, cloud is better :contentReference[oaicite:5]{index=5}
  const num_ctx = cloud ? 32000 : 8192;

  return ollama(modelName, {
    keep_alive: "10m",
    options: {
      num_ctx,
      repeat_penalty: 1.1,
    },
  });
}

// =====================================================
// 1) STREAMED TEXT (premium feel)
// =====================================================
export async function streamMatchupExplanation(payload: {
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  userQuestion: string;
}) {
  const model = getModel({ cloud: false });

  const result = await streamText({
    model,
    tools: { getPrediction },
    // prevent runaway tool loops
    stopWhen: stepCountIs(4),
    prompt: `
You are an NFL matchup analyst and coach.

Matchup:
${JSON.stringify({
  season: payload.season,
  week: payload.week,
  home_team: payload.home_team,
  away_team: payload.away_team,
})}

User question:
"${payload.userQuestion}"

Rules:
- Call getPrediction first.
- Explain clearly in 5-10 short sentences.
- Mention predicted score + win probability.
- Add 2-3 key reasons + 1 swing factor.
`,
  });

  return result.textStream;
}

// =====================================================
// 2) STRUCTURED JSON (UI card)
// =====================================================
export async function generateMatchupCard(payload: {
  season: number;
  week: number;
  home_team: string;
  away_team: string;
}) {
  const model = getModel({ cloud: false });

  const { output } = await generateText({
    model,
    tools: { getPrediction },
    output: Output.object({
      schema: MatchupCardSchema,
    }),
    stopWhen: stepCountIs(5),
    toolChoice: "auto",
    prompt: `
You are generating a clean UI card for an NFL prediction app.

Matchup:
${JSON.stringify(payload)}

Instructions:
- Call getPrediction for the matchup.
- Produce a concise headline.
- Bullets: 3 short reasons (max 5).
- predictedScore: integers.
- confidence: low/medium/high.
`,
  });

  return output;
}

type PredictWithNewsOptions = {
  baseline?: any;
  record?: boolean;
};

function mergeLlmPrediction(basePrediction: any, card: any) {
  const adjustedHome = clamp01(
    Number(
      card?.finalPick?.adjustedHomeWinProb ??
        basePrediction?.home_win_probability ??
        0.5
    )
  );

  // Keep baseline scores/metadata; only adjust win probabilities for LLM pick.
  return {
    ...basePrediction,
    home_win_probability: adjustedHome,
    away_win_probability: clamp01(1 - adjustedHome),
    prediction_source: "llm_adjusted",
    llm_card: card,
  };
}

// =====================================================
// 3) MODEL + NEWS (final pick)
// =====================================================
export async function predictWithNewsCard(
  input: MatchupInput,
  options: PredictWithNewsOptions = {}
) {
  const basePrediction =
    options.baseline ?? (await fetchPrediction(input, { record: options.record }));

  // Web search is optional and requires an API key; keep local-only by default.
  const canUseWeb = ENABLE_WEB_SEARCH && HAS_OLLAMA_API_KEY;

  // Use cloud model when web tools are enabled (larger context for citations).
  const model = getModel({ cloud: canUseWeb });
  // Only expose web tools here; baseline already supplied.
  // These tools are provided by ai-sdk-ollama and stay browser-safe.
  const tools = canUseWeb
    ? {
        webSearch: ollama.tools.webSearch({client:'ollama'
        }),
        webFetch: ollama.tools.webFetch({ maxContentLength: 5000 }),
      }
    : undefined;

  const { output } = await generateText({
    model,
    tools,
    output: Output.object({ schema: PredictWithNewsCardSchema }),
    stopWhen: stepCountIs(7),
    toolChoice: "auto",
    prompt: `
You are an NFL prediction analyst.

GOAL:
- Use the provided ML baseline from /api/predict.
- If webSearch/webFetch exist, research injuries/news that matter for THIS matchup.
- Then choose the winner and output a structured card.

BASELINE (from /api/predict):
${JSON.stringify(basePrediction)}

RULES:
1) Always start from BASELINE.modelSignals:
   - homeWinProb, pointDiff, predictedScore, total
2) If web tools exist, do:
   - webSearch: "<AWAY> vs <HOME> injury report", "<AWAY> vs <HOME> inactive", "<team> QB status"
   - webFetch: open 1-2 best sources
3) If news contradicts the ML baseline, reduce confidence and explain why.

OUTPUT REQUIREMENTS:
- baseModelSignals must mirror BASELINE.modelSignals.
- predictedScore should match ML score unless QB-out / major shock.
- adjustedHomeWinProb is your final belief (0..1).
- winner must match finalPick.lean.
- Provide 3-7 rationale bullets and 1-5 risk factors.
- Provide sources [{title,url}] based on webSearch results (2-6 max).

Matchup:
Season ${input.season}, Week ${input.week}
Away: ${input.away_team}
Home: ${input.home_team}
`,
  });

  return output;
}

export async function predictWithNewsPrediction(
  input: MatchupInput,
  options: PredictWithNewsOptions = {}
) {
  const baseline =
    options.baseline ?? (await fetchPrediction(input, { record: options.record }));
  const card = await predictWithNewsCard(input, { baseline });
  // Return a UnifiedPredictionResponse-compatible object for legacy UI consumers.
  return mergeLlmPrediction(baseline, card);
}
