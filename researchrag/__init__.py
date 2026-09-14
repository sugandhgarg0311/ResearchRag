"""Research RAG — research paper understanding, evidence retrieval, and implementation
reproduction.

The library is organized as an explicit, observable pipeline:

    parse → normalize → chunk → index → retrieve → fuse → rerank
          → assemble evidence → generate → validate

There is no hidden orchestration framework. Each stage is a small, typed,
unit-testable module (see ``PROJECT_ARCHITECTURE.md``).
"""

__version__ = "1.0.0"
