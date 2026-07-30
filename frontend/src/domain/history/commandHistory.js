import { eventBus } from '../../utils/eventBus';

/**
 * CommandHistory & Replay Engine - Records executed palette commands, debugger events, and IPC calls.
 */
class CommandHistory {
  constructor() {
    this.history = [];
  }

  logCommand(commandName, payload = {}, result = {}) {
    const entry = {
      id: `cmd_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
      command: commandName,
      payload,
      result,
      timestamp: new Date().toISOString(),
    };
    this.history.push(entry);
    eventBus.emit('history:entry', entry);
    return entry;
  }

  getHistory() {
    return [...this.history];
  }

  clear() {
    this.history = [];
  }
}

export const commandHistory = new CommandHistory();
