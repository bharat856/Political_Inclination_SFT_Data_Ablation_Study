#!/usr/bin/env python3
"""Prompt templates for Manifesto ideology benchmark v1."""

from __future__ import annotations


PROMPT_TEMPLATES = {
    "v1": """You are classifying a political manifesto excerpt. Choose the ideological orientation that best matches the political content of the excerpt. Return only one number: 1 for left, 2 for center, 3 for right.

Text:
{text}

Return only the number.""",
    "v2": """You are classifying a political manifesto excerpt.

Classify the local excerpt, not the party name.
Welfare, redistribution, public ownership, and labor rights usually lean left.
Market liberalization, tax cuts, law/order, nationalism, and traditional values usually lean right.
Mixed, technocratic, or moderate content can be center.

Return only 1, 2, or 3.

Text:
{text}

Return only the number.""",
}


def get_prompt_template(prompt_version: str) -> str:
    try:
        return PROMPT_TEMPLATES[prompt_version]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt version {prompt_version!r}. Available: {available}"
        ) from exc


def build_prompt(text: object, prompt_version: str = "v2") -> str:
    template = get_prompt_template(prompt_version)
    return template.format(text=str(text))
