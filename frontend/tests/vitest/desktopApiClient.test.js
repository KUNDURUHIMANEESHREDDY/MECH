import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, apiUrl } from '../../src/services/api';

afterEach(() => {
  delete window.appApi;
  vi.restoreAllMocks();
});

describe('Desktop OS API client', () => {
  it('uses the Vite same-origin proxy for browser API calls', () => {
    expect(apiUrl('/api/models')).toBe('/api/models');
  });

  it('uses the Electron local bridge for application logs when available', async () => {
    window.appApi = { getAppLogs: vi.fn().mockResolvedValue(['line one']) };
    await expect(api.getAppLogs()).resolves.toEqual(['line one']);
    expect(window.appApi.getAppLogs).toHaveBeenCalledOnce();
  });

  it('fails explicitly when a local-only capability has no bridge', async () => {
    await expect(api.getAppLogs()).rejects.toThrow(/Local application bridge unavailable/i);
  });
});
