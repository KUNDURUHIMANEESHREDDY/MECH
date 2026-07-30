import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import React from "react";
import CircuitExplorerPanel from "../../src/components/panels/CircuitExplorerPanel";
import CausalTraceViewerPanel from "../../src/components/panels/CausalTraceViewerPanel";
import FeatureAtlasPanel from "../../src/components/panels/FeatureAtlasPanel";
import InterventionTimelinePanel from "../../src/components/panels/InterventionTimelinePanel";
import TokenJourneyPanel from "../../src/components/panels/TokenJourneyPanel";
import EmbeddingViewerPanel from "../../src/components/panels/EmbeddingViewerPanel";
import PublicationExportModal from "../../src/components/PublicationExportModal";
import { exportService } from "../../src/services/exportService";

describe("Sprint 3 Visualization Panels & Export Engine", () => {
  it("renders CircuitExplorerPanel and selects nodes", () => {
    render(<CircuitExplorerPanel />);
    expect(screen.getByText(/Interactive Circuit Explorer/i)).toBeInTheDocument();
    const tokenNodes = screen.getAllByText(/The capital of France is/i);
    expect(tokenNodes.length).toBeGreaterThan(0);
    fireEvent.click(tokenNodes[1]);
    expect(screen.getByText(/Evidence & Provenance Overlay/i)).toBeInTheDocument();
  });

  it("renders CausalTraceViewerPanel and toggles playback", () => {
    render(<CausalTraceViewerPanel />);
    expect(screen.getByText(/Causal Trace Flow Viewer/i)).toBeInTheDocument();
    const playBtn = screen.getByText(/Play Flow ▶/i);
    fireEvent.click(playBtn);
    expect(screen.getByText(/Pause ⏸/i)).toBeInTheDocument();
  });

  it("renders FeatureAtlasPanel and filters features", () => {
    render(<FeatureAtlasPanel />);
    expect(screen.getByText(/SAE Feature Atlas/i)).toBeInTheDocument();
    const input = screen.getByPlaceholderText(/Filter features.../i);
    fireEvent.change(input, { target: { value: "Indirect" } });
    expect(screen.getByText(/Indirect Object Identifier/i)).toBeInTheDocument();
  });

  it("renders InterventionTimelinePanel", () => {
    render(<InterventionTimelinePanel />);
    expect(screen.getByText(/Intervention Timeline & Diff Viewer/i)).toBeInTheDocument();
    expect(screen.getByText(/Activation Patch/i)).toBeInTheDocument();
  });

  it("renders TokenJourneyPanel", () => {
    render(<TokenJourneyPanel />);
    expect(screen.getByText(/Token Representation Journey/i)).toBeInTheDocument();
    expect(screen.getByText(/Token \[0\]: "The"/i)).toBeInTheDocument();
  });

  it("renders EmbeddingViewerPanel and switches method", () => {
    render(<EmbeddingViewerPanel />);
    expect(screen.getByText(/3D Embedding Manifold Viewer/i)).toBeInTheDocument();
    const umapBtn = screen.getByText("UMAP");
    fireEvent.click(umapBtn);
    expect(screen.getByText(/Projection: UMAP/i)).toBeInTheDocument();
  });

  it("renders PublicationExportModal and generates export package", () => {
    render(<PublicationExportModal isOpen={true} onClose={vi.fn()} figureData={{ title: "Test Figure" }} />);
    expect(screen.getByText(/Automated Scientific Publication Engine/i)).toBeInTheDocument();
    const genBtn = screen.getByText(/Compile Replication Paper/i);
    fireEvent.click(genBtn);
    expect(screen.getByText(/✓ Generated paper.tex/i)).toBeInTheDocument();
  });

  it("exportService generates SVG and JSON artifacts deterministically", () => {
    const svgRes = exportService.exportSVG("My Analysis", [{ label: "Node 1" }]);
    expect(svgRes.filename).toBe("my_analysis_publication.svg");
    expect(svgRes.content).toContain("<svg");

    const jsonRes = exportService.exportJSON("My Analysis", { key: "val" });
    expect(jsonRes.filename).toBe("my_analysis_artifact.json");
    expect(jsonRes.content).toContain('"schemaVersion": "3.0.0"');
  });
});
