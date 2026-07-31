'use strict';

/**
 * Seed the local SQLite database with a few sample projects and recent
 * files so the UI has something to display on first launch.
 *
 * Run with: `npm run seed`
 */

const path = require('node:path');
const { getStorage } = require('../electron/storage');

async function main() {
  const storage = getStorage();
  await storage.init();

  const now = new Date().toISOString();
  const samples = [
    { id: 'prj_sample_1', name: 'Sample Project A', path: 'C:\\\\projects\\\\sample-a', createdAt: now, updatedAt: now },
    { id: 'prj_sample_2', name: 'Sample Project B', path: 'C:\\\\projects\\\\sample-b', createdAt: now, updatedAt: now }
  ];
  for (const p of samples) {
    await storage.upsert('projects', p);
  }

  const recents = [
    { id: 'rec_seed_1', path: 'C:\\\\docs\\\\overview.md', label: 'overview.md', openedAt: now },
    { id: 'rec_seed_2', path: 'C:\\\\docs\\\\notes.md', label: 'notes.md', openedAt: now }
  ];
  for (const r of recents) {
    await storage.upsert('recent_files', r);
  }

  console.log('Seeded', samples.length, 'projects and', recents.length, 'recent files.');
  storage.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
