#!/usr/bin/env node
// scripts/check_schedule.js
// Small E2E check: calls the dev-proxied schedule endpoint and prints results.
// Usage: node scripts/check_schedule.js

const URL = process.env.URL || 'http://localhost:3000/schedule/next-week';

console.log(`Checking schedule endpoint: ${URL}`);

(async () => {
  try {
    // Node 18+ has global fetch. If older Node, install node-fetch and uncomment the require.
    // const fetch = global.fetch || (await import('node-fetch')).default;
    const res = await fetch(URL, { method: 'GET' });
    console.log('Status:', res.status, res.statusText);
    const text = await res.text();
    try {
      const data = text ? JSON.parse(text) : null;
      if (Array.isArray(data)) {
        console.log('Returned array length:', data.length);
        console.log('Sample item (first):', JSON.stringify(data[0] || {}, null, 2));
      } else {
        console.log('Returned object/sample:', JSON.stringify(data, null, 2));
      }
    } catch (err) {
      console.log('Response is not JSON. Text preview:\n', text.slice(0, 1000));
    }
  } catch (err) {
    console.error('Fetch failed:', err && err.message ? err.message : err);
    process.exitCode = 2;
  }
})();
