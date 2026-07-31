import type { DesktopApi } from "../types";

export function getDesktopApi(): DesktopApi {
  if (!window.desktopApi) {
    throw new Error("Desktop API is unavailable. Launch the app through Electron.");
  }
  return window.desktopApi;
}
