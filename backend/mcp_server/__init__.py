"""MECH MCP control-plane package.

External AI agents control MECH through this MCP server, never through UI
automation. Every tool here calls the same Core API functions the Vue UI
calls (`backend.api.dispatcher`), so UI, MCP and CLI all drive one control
plane. Anything with no live executor (pause/resume, per-agent run) is
deliberately *absent* rather than simulated.
"""
