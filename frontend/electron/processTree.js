/**
 * Process Tree Management Utilities
 *
 * Provides cross-platform process tree termination for the Electron app.
 * These functions are used by the main process to clean up child processes
 * on shutdown, and by the dev script to kill spawned servers.
 */

const { execSync } = require('node:child_process');

/**
 * Terminate an entire process tree by PID.
 * On Windows, uses `taskkill /T /F` to recursively kill the tree.
 * On POSIX, sends SIGTERM to the process group (negative PID).
 *
 * @param {number} pid - Process ID of the root process to terminate
 * @param {Object} [logger] - Optional logger with `info` method
 */
function terminateProcessTree(pid, logger) {
  if (!pid) return;
  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /T /F /PID ${pid}`, { stdio: 'ignore' });
      if (logger) logger.info('process_tree_terminated', { pid, platform: 'win32' });
    } else {
      try {
        // Negative PID sends signal to the entire process group
        process.kill(-pid, 'SIGTERM');
      } catch {
        // Fallback: just the single process
        process.kill(pid, 'SIGTERM');
      }
      if (logger) logger.info('process_terminated', { pid, platform: process.platform });
    }
  } catch (err) {
    if (logger) logger.info('process_termination_skipped_already_exited', { pid, error: err.message });
  }
}

/**
 * Terminate a child process and its tree.
 * Wrapper around terminateProcessTree that extracts PID from a ChildProcess object.
 *
 * @param {ChildProcess} childProcess - The child process object from spawn/fork
 * @param {Object} [logger] - Optional logger
 */
function killProcessTree(childProcess, logger) {
  if (!childProcess || !childProcess.pid || childProcess.exitCode !== null) return;
  terminateProcessTree(childProcess.pid, logger);
}

module.exports = {
  terminateProcessTree,
  killProcessTree,
};