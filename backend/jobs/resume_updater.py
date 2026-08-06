"""Resume auto-update based on job descriptions."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("MECH.jobs.resume_updater")


# Keywords that signal graduation requirements
GRADUATION_KEYWORDS = [
    "bachelor", "bachelor's", "bs", "bsc", "b.sc", "beng", "b.tech",
    "master", "master's", "ms", "msc", "m.sc", "meng", "m.tech",
    "phd", "ph.d", "doctorate", "doctoral",
    "graduate", "graduation", "degree",
    "undergraduate", "undergrad",
    "university", "college", "institute",
]

SKILL_EXTRACTION_PATTERNS = [
    r"(?:required|must have|minimum|need(?:ing)?|looking for)\s+(?:[a-z]+\s+){0,3}(?:skills?|experience|expertise)[\s:]*([A-Za-z,\s]+?)(?:\.|$)",
    r"(?:we are looking for|we need|you should|candidate should|desired skills)[\s:]*([A-Za-z,\s]+?)(?:\.|$)",
    r"(?:proficiency|experience with|familiarity with|knowledge of)[\s:]*([A-Za-z,\s]+?)(?:\.|$)",
]


class ResumeUpdater:
    """Updates resume content to match a job description."""

    def __init__(self, resume_content: str, job_description: str):
        self.resume_content = resume_content
        self.job_description = job_description.lower()
        self.original_resume = resume_content

    def analyze_job_requirements(self) -> dict[str, Any]:
        """Extract key requirements from the job description."""
        requirements = {
            "required_skills": [],
            "required_experience": "",
            "education_required": "",
            "keywords": [],
            "graduation_required": False,
            "is_internship": False,
        }

        # Extract skills from common patterns
        for pattern in SKILL_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, self.job_description, re.IGNORECASE)
            for match in matches:
                skills = [s.strip() for s in match.split(",") if s.strip()]
                requirements["required_skills"].extend(skills)

        # Deduplicate skills
        requirements["required_skills"] = list(dict.fromkeys(requirements["required_skills"]))

        # Check for graduation requirement
        for keyword in GRADUATION_KEYWORDS:
            if keyword in self.job_description:
                requirements["graduation_required"] = True
                requirements["education_required"] = keyword
                break

        # Check if it's an internship
        internship_keywords = ["intern", "internship", "co-op", "cooperative", "trainee", "junior"]
        for kw in internship_keywords:
            if kw in self.job_description:
                requirements["is_internship"] = True
                break

        # Extract experience level
        exp_patterns = [
            r"(\d+)\s*(?:\+?\s*)?(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
            r"(?:minimum|min|at least)\s*(\d+)\s*(?:\+?\s*)?(?:years?|yrs?)",
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, self.job_description)
            if match:
                requirements["required_experience"] = match.group(1)
                break

        # Extract key skills/keywords from the full description
        # Common tech skills to look for
        all_skills = [
            "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
            "react", "angular", "vue", "svelte", "next.js", "node.js", "django",
            "flask", "fastapi", "spring", "dotnet", "ruby", "php", "swift", "kotlin",
            "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "ansible",
            "git", "ci/cd", "linux", "bash", "powershell", "machine learning",
            "deep learning", "nlp", "cv", "computer vision", "data science",
            "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "transformers",
            "graph neural networks", "gnn", "mechanistic interpretability",
            "attention", "transformer", "llm", "large language model",
            "research", "paper", "publication", "academic",
        ]
        found_skills = [s for s in all_skills if s.lower() in self.job_description]
        requirements["keywords"] = found_skills

        return requirements

    def update_resume(self) -> dict[str, Any]:
        """Update the resume to match the job description."""
        analysis = self.analyze_job_requirements()
        updated_resume = self.resume_content
        changes_made = []

        # If graduation is required and user is undergrad, add a note
        if analysis["graduation_required"]:
            grad_note = f"\n\n[Resume updated for {analysis['education_required'].title()} role]"
            if grad_note not in updated_resume:
                updated_resume += grad_note
                changes_made.append(f"Added graduation requirement note for {analysis['education_required']}")

        # Add job-relevant keywords to the skills section if not already present
        if analysis["required_skills"]:
            for skill in analysis["required_skills"]:
                if skill.lower() not in updated_resume.lower():
                    # Try to add to skills section
                    skills_section_pattern = r"(?i)(skills?[\s:]*\n)(.*?)(?=\n\n|\n#|\Z)"
                    match = re.search(skills_section_pattern, updated_resume, re.DOTALL)
                    if match:
                        existing = match.group(2)
                        if skill not in existing:
                            updated = existing.rstrip() + f", {skill}"
                            updated_resume = updated_resume.replace(existing, updated)
                            changes_made.append(f"Added '{skill}' to skills section")

        # Add keywords from the job description
        for keyword in analysis["keywords"]:
            if keyword.lower() not in updated_resume.lower():
                # Add to a relevant section or append
                changes_made.append(f"Note: keyword '{keyword}' found in job description")

        # Update experience section if required
        if analysis["required_experience"]:
            changes_made.append(f"Note: role requires {analysis['required_experience']}+ years experience")

        return {
            "updated_resume": updated_resume,
            "changes_made": changes_made,
            "analysis": analysis,
            "graduation_required": analysis["graduation_required"],
            "is_internship": analysis["is_internship"],
        }


def auto_update_resume(resume_content: str, job_description: str) -> dict[str, Any]:
    """Convenience function to auto-update a resume."""
    updater = ResumeUpdater(resume_content, job_description)
    return updater.update_resume()
