# MECH Platform: Third-Party Software & License Inventory

MECH Platform v2.0 utilizes the following open-source software packages:

---

## 1. Backend Dependencies (Python)

| Package | Version | License | Purpose |
| :--- | :--- | :--- | :--- |
| `torch` | 2.x | BSD-3-Clause | PyTorch Tensor & Deep Learning Runtime |
| `transformers` | 4.x | Apache-2.0 | Hugging Face Pretrained Transformer Architectures |
| `fastapi` | 0.110+ | MIT | REST API Server for Research Services |
| `uvicorn` | 0.28+ | BSD-3-Clause | ASGI Web Server Implementation |
| `pydantic` | 2.x | MIT | Data Validation & Settings Management |
| `psutil` | 5.9+ | BSD-3-Clause | Process & System Resource Monitoring |
| `pytest` | 8.x | MIT | Automated Testing Framework |

---

## 2. Frontend Dependencies (Node / Electron)

| Package | Version | License | Purpose |
| :--- | :--- | :--- | :--- |
| `electron` | 32.x | MIT | Cross-Platform Desktop Runtime |
| `react` | 18.x | MIT | User Interface Component Library |
| `react-dom` | 18.x | MIT | DOM Renderer for React |
| `better-sqlite3` | 11.x | MIT | High-Performance Embedded SQLite Storage |
| `lucide-react` | 0.400+ | ISC | Clean Scientific Iconography Suite |
| `zustand` | 4.5+ | MIT | Lightweight State Management Store |
| `vite` | 5.x | MIT | Fast Frontend Bundler & Build Tool |
| `vitest` | 2.x | MIT | Modern Unit Testing Runner |

---

## 3. Model Weight Licensing Notice

Model weights loaded by MECH (e.g. OpenAI GPT-2 weights via Hugging Face) are subject to their respective original licenses (e.g., Modified MIT License). MECH does not redistribute weights in its binary package; models are fetched dynamically to the user's local Hugging Face cache upon explicit user request.
