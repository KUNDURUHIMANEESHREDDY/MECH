/**
 * MECH Platform - Layout Manager
 * Manages the grid layout and responsive behavior of the application
 */

export interface LayoutConfig {
  activityBarWidth: number;
  sidebarWidth: number;
  sidebarCollapsedWidth: number;
  navbarHeight: number;
  statusBarHeight: number;
  consolePanelHeight: number;
}

export interface LayoutState {
  activityCollapsed: boolean;
  sidebarCollapsed: boolean;
  consoleVisible: boolean;
}

// Default layout configuration
export const defaultLayoutConfig: LayoutConfig = {
  activityBarWidth: 56,
  sidebarWidth: 280,
  sidebarCollapsedWidth: 64,
  navbarHeight: 48,
  statusBarHeight: 28,
  consolePanelHeight: 200,
};

// Generate CSS grid template based on layout state
export const generateGridTemplate = (
  layoutConfig: LayoutConfig,
  layoutState: LayoutState
): {
  gridTemplateColumns: string;
  gridTemplateRows: string;
} => {
  const activityWidth = layoutState.activityCollapsed ? '0px' : `${layoutConfig.activityBarWidth}px`;
  const sidebarWidth = layoutState.sidebarCollapsed ? `${layoutConfig.sidebarCollapsedWidth}px` : `${layoutConfig.sidebarWidth}px`;
  
  return {
    gridTemplateColumns: `${activityWidth} ${sidebarWidth} 1fr`,
    gridTemplateRows: `${layoutConfig.navbarHeight}px 1fr ${layoutConfig.statusBarHeight}px`,
  };
};

// Layout area definitions
export enum LayoutArea {
  ACTIVITY_BAR = 'activity-bar',
  SIDEBAR = 'sidebar',
  MAIN = 'main',
  TOPBAR = 'topbar',
  CONTENT = 'content',
  CONSOLE = 'console',
  STATUS_BAR = 'status-bar',
}

// Get grid area placement for each component
export const getGridPlacement = (
  area: LayoutArea,
  layoutState: LayoutState
): {
  gridColumn: string;
  gridRow: string;
} => {
  switch (area) {
    case LayoutArea.ACTIVITY_BAR:
      return {
        gridColumn: '1',
        gridRow: layoutState.consoleVisible ? '1 / -1' : '1 / -1',
      };
    case LayoutArea.SIDEBAR:
      return {
        gridColumn: '2',
        gridRow: '1 / -1',
      };
    case LayoutArea.MAIN:
      return {
        gridColumn: '3',
        gridRow: '1 / -1',
      };
    case LayoutArea.TOPBAR:
      return {
        gridColumn: layoutState.activityCollapsed ? '1 / -1' : '2 / -1',
        gridRow: '1',
      };
    case LayoutArea.CONTENT:
      return {
        gridColumn: '1 / -1',
        gridRow: '2',
      };
    case LayoutArea.CONSOLE:
      return {
        gridColumn: layoutState.activityCollapsed ? '1 / -1' : '2 / -1',
        gridRow: '3',
      };
    case LayoutArea.STATUS_BAR:
      return {
        gridColumn: '1 / -1',
        gridRow: '4',
      };
    default:
      return { gridColumn: '1', gridRow: '1' };
  }
};

// Responsive layout utilities
export const getResponsiveLayout = (
  width: number,
  layoutConfig: LayoutConfig
): LayoutState => {
  // On small screens, collapse both activity bar and sidebar
  if (width < 768) {
    return {
      activityCollapsed: true,
      sidebarCollapsed: true,
      consoleVisible: false,
    };
  }
  
  // On medium screens, collapse activity bar
  if (width < 1024) {
    return {
      activityCollapsed: true,
      sidebarCollapsed: false,
      consoleVisible: false,
    };
  }
  
  // On large screens, show everything
  return {
    activityCollapsed: false,
    sidebarCollapsed: false,
    consoleVisible: true,
  };
};

// Layout context for React components
import { createContext, useContext, useState, useEffect } from 'react';

interface LayoutContextType {
  config: LayoutConfig;
  state: LayoutState;
  setState: (state: LayoutState) => void;
  toggleActivity: () => void;
  toggleSidebar: () => void;
  toggleConsole: () => void;
}

const LayoutContext = createContext<LayoutContextType | undefined>(undefined);

export const LayoutProvider: React.FC<{
  children: React.ReactNode;
  config?: Partial<LayoutConfig>;
  initialState?: Partial<LayoutState>;
}> = ({
  children,
  config: configOverrides = {},
  initialState: initialStateOverrides = {},
}) => {
  const [state, setState] = useState<LayoutState>({
    activityCollapsed: false,
    sidebarCollapsed: false,
    consoleVisible: true,
    ...initialStateOverrides,
  });

  const config: LayoutConfig = {
    ...defaultLayoutConfig,
    ...configOverrides,
  };

  const toggleActivity = () => {
    setState(prev => ({ ...prev, activityCollapsed: !prev.activityCollapsed }));
  };

  const toggleSidebar = () => {
    setState(prev => ({ ...prev, sidebarCollapsed: !prev.sidebarCollapsed }));
  };

  const toggleConsole = () => {
    setState(prev => ({ ...prev, consoleVisible: !prev.consoleVisible }));
  };

  // Handle window resize for responsive layout
  useEffect(() => {
    const handleResize = () => {
      const responsiveState = getResponsiveLayout(window.innerWidth, config);
      setState(prev => ({
        ...prev,
        ...responsiveState,
      }));
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, [config]);

  return (
    <LayoutContext.Provider
      value={{
        config,
        state,
        setState,
        toggleActivity,
        toggleSidebar,
        toggleConsole,
      }}
    >
      {children}
    </LayoutContext.Provider>
  );
};

export const useLayout = (): LayoutContextType => {
  const context = useContext(LayoutContext);
  if (!context) {
    throw new Error('useLayout must be used within a LayoutProvider');
  }
  return context;
};

export default {
  defaultLayoutConfig,
  generateGridTemplate,
  getGridPlacement,
  getResponsiveLayout,
  LayoutProvider,
  useLayout,
};
