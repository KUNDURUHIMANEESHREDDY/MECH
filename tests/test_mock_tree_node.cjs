'use strict';
/**
 * Adversarial test harness for Windows process tree termination in Node.js.
 * Imports and exercises the real production functions from frontend/electron/processTree.js
 */

const { spawn, execSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const processTree = require('../frontend/electron/processTree.js');

const { terminateProcessTree, killProcessTree } = processTree;

const REGISTRY_FILE = path.join(__dirname, 'mock_tree_pids.json');


function isPidAlive(pid) {
  try {
    // On Windows, process.kill(pid, 0) tests existence without signaling
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return false;
  }
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runTest() {
  console.log('=== TEST 1: Multi-level Node -> Python Process Tree Termination ===');
  
  if (fs.existsSync(REGISTRY_FILE)) {
    fs.unlinkSync(REGISTRY_FILE);
  }

  // Helper mock script that spawns children and records PIDs
  // Level 1: Root process spawns Level 2 child, which spawns Level 3 grandchild
  const mockChildScript = `
import sys, time, subprocess, json, os

my_pid = os.getpid()
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 1
reg_file = sys.argv[2]

# Append PID to registry
try:
    with open(reg_file, 'r') as f:
        data = json.load(f)
except Exception:
    data = []
data.append({'depth': depth, 'pid': my_pid})
with open(reg_file, 'w') as f:
    json.dump(data, f)

if depth < 3:
    # Spawn child at depth + 1
    child = subprocess.Popen([sys.executable, '-c', sys.argv[3], str(depth + 1), reg_file, sys.argv[3]])
    child.wait()
else:
    time.sleep(60)
`;

  // Start Root Process (Level 1)
  const rootChild = spawn('python', [
    '-c', mockChildScript,
    '1', REGISTRY_FILE, mockChildScript
  ], { stdio: 'ignore' });

  console.log(`Root process spawned with PID: ${rootChild.pid}`);

  // Wait for the full 3-level tree to register
  let pids = [];
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    if (fs.existsSync(REGISTRY_FILE)) {
      try {
        pids = JSON.parse(fs.readFileSync(REGISTRY_FILE, 'utf8'));
        if (pids.length >= 3) break;
      } catch (e) {}
    }
  }

  console.log(`Discovered process tree members (${pids.length} processes):`);
  pids.forEach(p => console.log(`  - Depth ${p.depth}: PID ${p.pid} (alive=${isPidAlive(p.pid)})`));

  if (pids.length < 3) {
    throw new Error(`Failed to build full 3-level tree, only got ${pids.length} processes`);
  }

  // Verify all are alive before termination
  for (const p of pids) {
    if (!isPidAlive(p.pid)) {
      throw new Error(`Process PID ${p.pid} at depth ${p.depth} died prematurely!`);
    }
  }
  console.log('[PASS] All 3 process tree levels confirmed alive simultaneously.');

  // Terminate using terminateProcessTree on Root PID
  console.log(`\nInvoking terminateProcessTree(${rootChild.pid})...`);
  const mockLogger = {
    info: (msg, meta) => console.log(`  [logger.info] ${msg}:`, JSON.stringify(meta)),
    warn: (msg, meta) => console.log(`  [logger.warn] ${msg}:`, JSON.stringify(meta)),
  };
  terminateProcessTree(rootChild.pid, mockLogger);

  await sleep(1000);

  // Verify that EVERY process in the tree is terminated
  let survivorCount = 0;
  for (const p of pids) {
    const alive = isPidAlive(p.pid);
    console.log(`Post-kill check PID ${p.pid} (Depth ${p.depth}): ${alive ? 'ALIVE (ORPHAN!)' : 'TERMINATED (PASS)'}`);
    if (alive) {
      survivorCount++;
      // Clean up orphan forcefully
      try { execSync(`taskkill /F /PID ${p.pid}`, { stdio: 'ignore' }); } catch {}
    }
  }

  if (survivorCount > 0) {
    throw new Error(`Process tree termination failed! ${survivorCount} orphan processes survived!`);
  }
  console.log('[PASS] Complete process tree cleanly eliminated with 0 orphans.');

  console.log('\n=== TEST 2: killProcessTree Edge Cases ===');
  // 1. Dead process
  console.log('Testing dead PID:');
  terminateProcessTree(999999, mockLogger);
  console.log('[PASS] Handled non-existent/already dead PID without throwing uncaught error.');

  // 2. Null/undefined childProcess
  killProcessTree(null, mockLogger);
  killProcessTree({}, mockLogger);
  killProcessTree({ pid: null }, mockLogger);
  console.log('[PASS] Handled null/undefined childProcess objects safely.');

  // 3. killProcessTree with valid child process object
  const dummyChild = { pid: 12345, exitCode: null };
  killProcessTree(dummyChild, mockLogger);
  console.log('[PASS] killProcessTree handles valid child process object.');

  // Clean up test file
  try { fs.unlinkSync(REGISTRY_FILE); } catch {}
  console.log('\n=== ALL WINDOWS PROCESS TREE TESTS PASSED! ===');
}

runTest().catch((err) => {
  console.error('\n[FATAL TEST FAILURE]:', err);
  process.exit(1);
});