import { create } from 'zustand';

export interface UIState {
  commandPaletteOpen: boolean;
  activityCollapsed: boolean;
  sidebarCollapsed: boolean;
  crumb: string;
}

export type UIStore = UIState & {
  toggleCommandPalette: () => void;
  setCommandPaletteOpen: (open: boolean) => void;
  toggleActivity: () => void;
  toggleSidebar: () => void;
  setCrumb: (crumb: string) => void;
};

export const useUIStore = create<UIStore>((set) => ({
  commandPaletteOpen: false,
  activityCollapsed: false,
  sidebarCollapsed: true,
  crumb: 'MECH',

  toggleCommandPalette: () =>
    set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),
  setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
  toggleActivity: () =>
    set((state) => ({ activityCollapsed: !state.activityCollapsed })),
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setCrumb: (crumb) => set({ crumb }),
}));
