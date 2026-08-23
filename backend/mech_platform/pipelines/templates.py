"""Backward-compatible re-export of PipelineTemplates from backend.research_platform.pipelines.templates."""

from backend.research_platform.pipelines.templates import (
    PipelineTemplate,
    get_template,
    list_templates,
)

__all__ = ["PipelineTemplate", "get_template", "list_templates"]
