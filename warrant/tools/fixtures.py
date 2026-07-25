"""Offline fixtures so tools (and the whole pipeline) run without a network.

These are lightly-condensed, real papers drawn from the AI_Daily_Brief corpus,
so the offline demo produces a genuine-looking briefing with real metrics that
the grounding guard can verify against the source text.
"""

from __future__ import annotations

PAPERS: dict[str, dict] = {
    "2603.26993": {
        "arxiv_id": "2603.26993",
        "title": "On the Reliability Limits of LLM-Based Multi-Agent Planning",
        "authors": ["Ruicheng Ao", "Siyang Gao", "David Simchi-Levi"],
        "published": "2026-03-27",
        "categories": ["cs.MA", "cs.AI"],
        "abstract": (
            "We model an LLM multi-agent system as a finite acyclic delegated "
            "decision network and prove that, without new exogenous signals, any "
            "delegated network is dominated by a centralized Bayes decision maker "
            "with the same information. Communication loss equals posterior "
            "distortion. Empirically, relay chains without new signals collapse "
            "accuracy from 90.7% to 22.5% at five stages."
        ),
        "pdf_text": (
            "On the Reliability Limits of LLM-Based Multi-Agent Planning.\n"
            "We model any LLM agent system as a finite acyclic directed graph "
            "G=(V,E) of decision stages. Each node observes a common signal B or a "
            "private exogenous signal Z_v (a tool call result, database lookup, or "
            "environment observation). The terminal node makes the final decision.\n"
            "Main theorem: without new exogenous signals, any delegated multi-agent "
            "network is decision-theoretically dominated by a single centralized "
            "Bayes decision maker with access to the same information. Added roles "
            "built from the same model over overlapping context merely reorganize "
            "the same evidence.\n"
            "Communication loss equals posterior distortion. Under the logarithmic "
            "scoring rule the loss equals conditional mutual information I(Y;H|M); "
            "under the Brier score it equals expected squared posterior error.\n"
            "Experiments on 200 MMLU four-way questions with gpt-4.1-mini and "
            "o4-mini across nine conditions. A single centralized agent scores "
            "90.7% accuracy. A two-stage prose relay drops to 41.2%, three stages "
            "to 43.5%, and five stages to 22.5%, which is below the 25% random "
            "baseline. A structured posterior-vector message format degrades only "
            "2.8 points per stage versus 8.5 points for prose relay. Per-question "
            "KL divergence between agent posteriors predicts the accuracy drop with "
            "correlation r=0.72. Adding a Wikipedia tool to MMLU changes accuracy "
            "from 90.7% to 87.2%, a slight decrease, because the model already "
            "knows the answers. Adding a synthetic knowledge-base lookup for facts "
            "the model cannot know raises accuracy from 24.3% to 82.7%, a 58.4 "
            "point gain. Escalate to human review when automated posterior risk "
            "R_a(H) exceeds review loss R_h(H), not by fixed step counts."
        ),
        # Verifiable numeric facts the grounding oracle checks against the draft.
        "key_metrics": ["90.7%", "41.2%", "22.5%", "2.8", "8.5", "r=0.72", "82.7%", "58.4"],
    },
    "2511.13646": {
        "arxiv_id": "2511.13646",
        "title": "Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?",
        "authors": ["Chunqiu Steven Xia", "Yifeng Wang", "Yuxiang Wei", "Lingming Zhang"],
        "published": "2025-11-17",
        "categories": ["cs.SE", "cs.AI"],
        "abstract": (
            "Live-SWE-agent is the first live software agent that autonomously "
            "evolves itself on the fly during runtime, starting from a minimal "
            "bash-only scaffold with no offline training. It synthesizes custom "
            "tools mid-task and achieves 77.4% on SWE-bench Verified."
        ),
        "pdf_text": (
            "Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the "
            "Fly? Software agents are themselves software, so they can self-modify "
            "at runtime. Live-SWE-agent starts from a minimal bash-only scaffold "
            "(mini-SWE-agent, about 100 lines) with no offline training. At each "
            "step the agent may output a command or synthesize a custom tool. A "
            "reflection message after each environmental feedback prompts the agent "
            "to decide whether to create or revise a tool.\n"
            "Results: 77.4% resolve rate on SWE-bench Verified and 45.8% on "
            "SWE-Bench Pro, both without test-time scaling. On SWE-bench "
            "Verified-60, it reaches 65.0% versus DGM's 53.3%, at zero offline "
            "cost, whereas a single DGM run costs around $22,000. Per-issue cost "
            "stayed near $0.50."
        ),
        "key_metrics": ["77.4%", "45.8%", "65.0%", "53.3%", "$22,000", "$0.50"],
    },
}

# Newest-video fixture for the YouTube discovery tool.
YOUTUBE_LATEST: dict[str, dict] = {
    "last-week-in-ai": {
        "video_id": "lwia-2026-07-25",
        "title": "Last Week in AI #217 — Multi-Agent Reliability, Self-Evolving Agents",
        "url": "https://www.youtube.com/watch?v=lwia-2026-07-25",
        "description": (
            "This week we discuss reliability limits of multi-agent planning "
            "(arXiv:2603.26993) and self-evolving software agents "
            "(arXiv:2511.13646). Papers mentioned: 2603.26993, 2511.13646."
        ),
        "arxiv_ids": ["2603.26993", "2511.13646"],
    }
}
