// Domain-level API helpers.
// Kept as a thin wrapper so existing imports like `./api/nfl.js` keep working.

export { health, getNextWeekSchedule, predictGame } from "./client.js";
