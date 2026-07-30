# Distributed Runtime Documentation (Sprint 3)

Details hardware target abstractions, multi-GPU partitioning, layer streaming, and compression.

---

## Capabilities
- **Execution Targets**: `LocalCPU`, `LocalGPU`, `RemoteGPU`, `Cluster`.
- **Multi-GPU Partitioning**: `DataParallel`, `TensorParallel`, `PipelineParallel`.
- **Weight Streaming**: On-demand VRAM layer streaming.
- **Activation Compression**: `FP16` and `INT8` activation codecs.
