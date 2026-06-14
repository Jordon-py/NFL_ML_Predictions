/**
 * Shared normalization helpers for "game-like" objects used across the frontend.
 *
 * Why this exists:
 * - schedule rows, prediction responses, and history entries are close but not identical
 * - several screens need to derive the same matchup key and API payload
 * - keeping that logic in one place reduces lookup bugs and makes the data flow easier to teach
 */

function toNumberOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

const TEAM_CODE_ALIASES = {
  LA: "LAR",
  STL: "LAR",
  SD: "LAC",
  OAK: "LV",
  WSH: "WAS",
};

/**
 * Normalize a team identifier into the uppercase code style expected by the API.
 *
 * The backend is most reliable when home/away teams are sent in a stable format,
 * so the UI normalizes early and then reuses that value everywhere.
 */
export function normalizeTeamCode(value) {
  const code = (value ?? "").toString().trim().toUpperCase();
  return TEAM_CODE_ALIASES[code] || code;
}

/**
 * Read season from whichever schedule shape we received.
 * Some older payloads use `season_num`.
 */
export function getGameSeason(gameLike) {
  return toNumberOrNull(gameLike?.season ?? gameLike?.season_num);
}

/**
 * Read week from whichever schedule shape we received.
 * Some older payloads use `week_num` or `week_number`.
 */
export function getGameWeek(gameLike) {
  return toNumberOrNull(
    gameLike?.week ?? gameLike?.week_num ?? gameLike?.week_number
  );
}

/**
 * Build the canonical lookup key used by Dashboard, TeamGrid, and StatsPage.
 *
 * Important:
 * This intentionally stays independent from `game_id` because the UI already
 * depends on the season-week-home-away composite format in several places.
 */
export function buildMatchupKey(gameLike) {
  const season = getGameSeason(gameLike) ?? "";
  const week = getGameWeek(gameLike) ?? "";
  const home = normalizeTeamCode(gameLike?.home_abbr ?? gameLike?.home_team);
  const away = normalizeTeamCode(gameLike?.away_abbr ?? gameLike?.away_team);
  return [season, week, home, away].filter(Boolean).join("-");
}

/**
 * Build the minimal payload required by `POST /predict`.
 */
export function buildPredictPayload(gameLike) {
  return {
    home_team: normalizeTeamCode(gameLike?.home_abbr ?? gameLike?.home_team),
    away_team: normalizeTeamCode(gameLike?.away_abbr ?? gameLike?.away_team),
    season: getGameSeason(gameLike),
    week: getGameWeek(gameLike),
  };
}

/**
 * Convert a raw schedule row into the smaller shape expected by the card UI.
 *
 * This keeps presentational components focused on rendering instead of
 * repeatedly decoding backend field aliases.
 */
export function normalizeMatchup(gameLike = {}) {
  const homeTeam = normalizeTeamCode(gameLike?.home_abbr ?? gameLike?.home_team);
  const awayTeam = normalizeTeamCode(gameLike?.away_abbr ?? gameLike?.away_team);

  return {
    ...gameLike,
    season: getGameSeason(gameLike),
    week: getGameWeek(gameLike),
    home_team: homeTeam,
    away_team: awayTeam,
    home_abbr: homeTeam || gameLike?.home_abbr,
    away_abbr: awayTeam || gameLike?.away_abbr,
    home_logo: gameLike?.home_logo ?? null,
    away_logo: gameLike?.away_logo ?? null,
    kickoff: gameLike?.kickoff ?? null,
  };
}
