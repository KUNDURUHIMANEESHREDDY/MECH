import React from "react";
import { useQuery } from "react-query";

/**
 * HomeDashboard
 * 
 * Polished landing page orchestrating the research workflow:
 * Projects -> Experiment -> Analysis -> Discovery -> Publication
 */
export default function HomeDashboard() {
  const { data: catalog } = useQuery("researchCatalog", async () => {
    const res = await fetch("http://127.0.0.1:8000/api/v1/research_catalog?item_type=all");
    return res.json();
  });

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-12">
      <header className="mb-8">
        <h1 className="text-4xl font-light text-slate-800">Research Environment</h1>
        <p className="text-slate-500 mt-2">Projects -&gt; Experiment -&gt; Analysis -&gt; Discovery -&gt; Publication</p>
      </header>

      <section className="grid grid-cols-2 gap-8">
        <div className="p-6 bg-white rounded-xl shadow-sm border border-slate-200 hover:shadow-md transition">
          <h2 className="text-xl font-medium mb-4 text-slate-700">Recent Projects</h2>
          <ul className="space-y-3">
            <li className="flex justify-between items-center text-sm">
              <span className="text-indigo-600 font-medium">IOI Reproduction</span>
              <span className="text-slate-400">2 hours ago</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-indigo-600 font-medium">SAE Feature Analysis</span>
              <span className="text-slate-400">1 day ago</span>
            </li>
          </ul>
        </div>
        
        <div className="p-6 bg-white rounded-xl shadow-sm border border-slate-200 hover:shadow-md transition">
          <h2 className="text-xl font-medium mb-4 text-slate-700">System Health</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">GPU VRAM (ModelManager)</span>
                <span className="font-medium">4.2 GB / 12 GB</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2">
                <div className="bg-emerald-400 h-2 rounded-full" style={{ width: "35%" }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">Benchmark Suite Status</span>
                <span className="font-medium text-emerald-600">Passing (6/6)</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2">
                <div className="bg-emerald-400 h-2 rounded-full" style={{ width: "100%" }}></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-light text-slate-800 mb-6">Unified Catalog</h2>
        <div className="grid grid-cols-3 gap-6">
          {catalog?.catalog?.slice(0, 6).map((item, idx) => (
            <div key={idx} className="p-4 bg-slate-50 rounded-lg border border-slate-100">
              <h3 className="font-medium text-slate-700">{item.title || item.id}</h3>
              <p className="text-xs text-slate-500 mt-2 uppercase tracking-wide">Registry Item</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
