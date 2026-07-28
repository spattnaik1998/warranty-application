"""Source ingestion: bring a repo's Python files in for a static scan.

The only thing the static analyzer needs is ``{relative_path: source}``; this
package produces that dict from a GitHub URL, an ``owner/repo`` slug, or a local
directory, without executing any of it.
"""

from warrant.ingest.github import (
    RepoRef,
    collect_python_sources,
    resolve_target,
)

__all__ = ["RepoRef", "collect_python_sources", "resolve_target"]
