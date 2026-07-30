import React, { useState } from "react";
import { extensionMarketplaceService } from "../services/extensionMarketplaceService";

export default function ExtensionMarketplaceModal({ isOpen, onClose }) {
  const [plugins, setPlugins] = useState(extensionMarketplaceService.listAvailablePlugins());
  const [installed, setInstalled] = useState(extensionMarketplaceService.listInstalledPlugins());

  if (!isOpen) return null;

  const handleInstall = (pluginId) => {
    extensionMarketplaceService.installPlugin(pluginId);
    setInstalled(extensionMarketplaceService.listInstalledPlugins());
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-lg w-full p-5 text-slate-100 shadow-2xl">
        <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
          <h3 className="text-base font-bold text-amber-400">🧩 Extension Marketplace</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-sm">
            ✕
          </button>
        </div>

        <div className="space-y-3 max-h-80 overflow-y-auto">
          {plugins.map((plug) => {
            const isInst = installed.some((p) => p.id === plug.id);
            return (
              <div key={plug.id} className="p-3 bg-slate-800/80 rounded border border-slate-700 flex justify-between items-center">
                <div>
                  <h4 className="text-xs font-bold text-slate-200">{plug.name}</h4>
                  <p className="text-[10px] text-slate-400">By {plug.author} • v{plug.version}</p>
                </div>
                <button
                  disabled={isInst}
                  onClick={() => handleInstall(plug.id)}
                  className={`px-3 py-1 text-xs font-semibold rounded transition ${
                    isInst
                      ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                      : "bg-amber-600 hover:bg-amber-500 text-white"
                  }`}
                >
                  {isInst ? "Installed ✓" : "Install Plugin"}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
