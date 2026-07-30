import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import React from "react";
import VersionControlPanel from "../../src/components/panels/VersionControlPanel";
import ArtifactManagerPanel from "../../src/components/panels/ArtifactManagerPanel";
import PublicationBuilderPanel from "../../src/components/panels/PublicationBuilderPanel";
import WorkspaceSharingModal from "../../src/components/WorkspaceSharingModal";
import ExtensionMarketplaceModal from "../../src/components/ExtensionMarketplaceModal";
import { collaborationManager } from "../../src/services/collaborationManager";
import { experimentVersionControl } from "../../src/services/experimentVersionControl";
import { artifactManagerService } from "../../src/services/artifactManagerService";
import { extensionMarketplaceService } from "../../src/services/extensionMarketplaceService";
import { researchTemplateManager } from "../../src/services/researchTemplateManager";

describe("Sprint 3 AI 5 IDE & Infrastructure Platform", () => {
  it("renders VersionControlPanel and commits state snapshot", () => {
    render(<VersionControlPanel />);
    expect(screen.getByText(/Experiment Version Control/i)).toBeInTheDocument();
    const input = screen.getByPlaceholderText(/Commit message/i);
    fireEvent.change(input, { target: { value: "Patched Layer 8 Neuron 402" } });
    const btn = screen.getByText(/Commit Snapshot/i);
    fireEvent.click(btn);
    expect(screen.getByText("Patched Layer 8 Neuron 402")).toBeInTheDocument();
  });

  it("renders ArtifactManagerPanel and lists managed research artifacts", () => {
    render(<ArtifactManagerPanel />);
    expect(screen.getByText(/Managed Research Artifacts/i)).toBeInTheDocument();
    expect(screen.getByText(/Circuit Graph Figure 1/i)).toBeInTheDocument();
  });

  it("renders PublicationBuilderPanel and recompiles manuscript", () => {
    render(<PublicationBuilderPanel />);
    expect(screen.getByText(/Publication Manuscript Builder/i)).toBeInTheDocument();
    const btn = screen.getByText(/Recompile Paper ⚡/i);
    fireEvent.click(btn);
    expect(screen.getByText(/Updated Mechanistic Paper/i)).toBeInTheDocument();
  });

  it("renders WorkspaceSharingModal and generates share URL", () => {
    render(<WorkspaceSharingModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText(/Collaborative Workspace Sharing/i)).toBeInTheDocument();
    const genBtn = screen.getByText(/Generate Shareable Workspace URL & Manifest/i);
    fireEvent.click(genBtn);
    expect(screen.getByDisplayValue(/https:\/\/antigravity.research\/ws\//i)).toBeInTheDocument();
  });

  it("renders ExtensionMarketplaceModal and installs plugin", () => {
    render(<ExtensionMarketplaceModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText(/Extension Marketplace/i)).toBeInTheDocument();
    const installBtns = screen.getAllByText(/Install Plugin/i);
    fireEvent.click(installBtns[0]);
    expect(screen.getByText(/TransformerLens Bridge/i)).toBeInTheDocument();
  });

  it("researchTemplateManager instantiates workflow templates", () => {
    const tpls = researchTemplateManager.listTemplates();
    expect(tpls.length).toBeGreaterThan(0);
    const inst = researchTemplateManager.instantiateTemplate("template_circuit_discovery");
    expect(inst.status).toBe("initialized");
  });
});
