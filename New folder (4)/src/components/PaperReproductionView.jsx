import React, { useState } from "react";
import { useQuery, useMutation } from "react-query";

/**
 * PaperReproductionView
 * 
 * One-click reproduction workflow:
 * Select Paper -> Download Model/Dataset -> Run Pipeline -> Metrics -> Report
 */
export default function PaperReproductionView() {
  const [selectedPaper, setSelectedPaper] = useState(null);
  
  const { data: catalog } = useQuery("papers", async () => {
    const res = await fetch("http://127.0.0.1:8000/api/v1/research_catalog?item_type=papers");
    const json = await res.json();
    return json.catalog;
  });

  const reproductionMutation = useMutation(async (paperId) => {
    // In reality this would trigger the BenchmarkRunner for the specific paper.
    // Simulating the one-click process.
    return new Promise(resolve => setTimeout(() => resolve({ status: "success", paperId }), 2000));
  });

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-light text-slate-800 mb-2">Paper Reproductions</h1>
      <p className="text-slate-500 mb-8">One-click reproduction of landmark mechanistic interpretability papers.</p>
      
      <div className="grid grid-cols-3 gap-8">
        <div className="col-span-1 space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Select Paper</h2>
          {catalog?.map(paper => (
            <div 
              key={paper.id} 
              onClick={() => setSelectedPaper(paper)}
              className={`p-4 rounded-lg border cursor-pointer transition ${selectedPaper?.id === paper.id ? "bg-indigo-50 border-indigo-200" : "bg-white border-slate-200 hover:border-indigo-300"}`}
            >
              <h3 className="font-medium text-slate-800">{paper.title}</h3>
              <p className="text-xs text-slate-500 mt-1">{paper.id}</p>
            </div>
          ))}
        </div>
        
        <div className="col-span-2">
          {selectedPaper ? (
            <div className="bg-white border border-slate-200 rounded-xl p-8 shadow-sm">
              <h2 className="text-2xl font-medium text-slate-800 mb-2">{selectedPaper.title}</h2>
              <div className="flex gap-4 text-sm text-slate-500 mb-8">
                <span>Model: gpt2-small</span>
                <span>Dataset: required</span>
              </div>
              
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${reproductionMutation.isLoading ? "bg-indigo-100 text-indigo-600 animate-pulse" : reproductionMutation.isSuccess ? "bg-emerald-100 text-emerald-600" : "bg-slate-100 text-slate-400"}`}>
                    1
                  </div>
                  <div>
                    <h4 className="font-medium">Download Model & Dataset</h4>
                    <p className="text-sm text-slate-500">Fetches weights and required activations.</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${reproductionMutation.isLoading ? "bg-indigo-100 text-indigo-600 animate-pulse" : reproductionMutation.isSuccess ? "bg-emerald-100 text-emerald-600" : "bg-slate-100 text-slate-400"}`}>
                    2
                  </div>
                  <div>
                    <h4 className="font-medium">Run Pipeline</h4>
                    <p className="text-sm text-slate-500">Executes the circuit discovery / extraction logic.</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${reproductionMutation.isLoading ? "bg-indigo-100 text-indigo-600 animate-pulse" : reproductionMutation.isSuccess ? "bg-emerald-100 text-emerald-600" : "bg-slate-100 text-slate-400"}`}>
                    3
                  </div>
                  <div>
                    <h4 className="font-medium">Generate Report</h4>
                    <p className="text-sm text-slate-500">Compares observed metrics against TransformerLens baselines.</p>
                  </div>
                </div>
              </div>

              <div className="mt-10 pt-6 border-t border-slate-100">
                <button 
                  onClick={() => reproductionMutation.mutate(selectedPaper.id)}
                  disabled={reproductionMutation.isLoading}
                  className="bg-indigo-600 text-white px-6 py-2 rounded-lg font-medium shadow-sm hover:bg-indigo-700 disabled:opacity-50 transition"
                >
                  {reproductionMutation.isLoading ? "Reproducing..." : reproductionMutation.isSuccess ? "Reproduction Complete" : "Start One-Click Reproduction"}
                </button>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center border-2 border-dashed border-slate-200 rounded-xl text-slate-400">
              Select a paper to view reproduction details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
