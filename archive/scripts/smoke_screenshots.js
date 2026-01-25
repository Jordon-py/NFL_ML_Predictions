const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const url = process.argv[2] || 'http://localhost:3000';
  const viewports = [
    { name: 'mobile', width: 375, height: 800 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'desktop', width: 1366, height: 900 }
  ];

  if (!fs.existsSync('screenshots')) fs.mkdirSync('screenshots');

  console.log(`Opening ${url}`);
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  try {
    const page = await browser.newPage();
    for (const vp of viewports) {
      console.log(`Rendering ${vp.name} ${vp.width}x${vp.height}...`);
      await page.setViewport({ width: vp.width, height: vp.height });
      await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
      // small delay to let animations settle
      await page.waitForTimeout(600);
      const path = `screenshots/${vp.name}-${vp.width}x${vp.height}.png`;
      await page.screenshot({ path, fullPage: false });
      console.log(`Saved ${path}`);
    }
  } catch (err) {
    console.error('Error capturing screenshots:', err);
    process.exitCode = 2;
  } finally {
    await browser.close();
  }
})();
