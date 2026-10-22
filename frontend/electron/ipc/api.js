'use strict';

/**
 * The renderer's only route to the control plane.
 *
 * `api:request` is the authenticated proxy: the main process validates the
 * caller's frame, validates the request shape, attaches the bearer token and
 * returns `{ status, body }`. The token itself never crosses the bridge.
 */

const { apiRequest, isTrustedRendererUrl } = require('../apiProxy');

/**
 * Refuse a request that did not come from our own renderer.
 *
 * `senderFrame.url` is set by Electron, not by the page, so a compromised
 * renderer cannot claim a different origin here.
 */
function assertTrustedSender(event) {
  const url = event && event.senderFrame ? event.senderFrame.url : '';
  if (!isTrustedRendererUrl(url)) {
    throw new Error(`rejected api request from untrusted renderer: ${url || '<unknown>'}`);
  }
}

function registerApiHandlers({ ipcMain, logger }) {
  ipcMain.handle('api:request', async (event, request) => {
    assertTrustedSender(event);
    try {
      return await apiRequest(request);
    } catch (err) {
      logger.error('api_proxy_error', {
        error: err.message,
        method: request && request.method,
        path: request && request.path,
      });
      throw new Error(err.message);
    }
  });
}

module.exports = { assertTrustedSender, registerApiHandlers };
