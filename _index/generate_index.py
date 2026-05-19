#!/usr/bin/env python3
"""
generate_index.py — Paper Lens 索引生成腳本

掃描 topics/**/meta.yaml,依以下維度生成索引檔案:
- _index/by-date.md     依論文發表時間
- _index/by-topic.md    依主題分類
- _index/by-paper.md    依論文反查
- _index/reading-map.md 論文依賴圖(Mermaid)

使用方式:
    python scripts/generate_index.py              # 寫入 _index/
    python scripts/generate_index.py --dry-run    # 只印到 stdout
    python scripts/generate_index.py --check      # 檢查索引是否需要更新,需要時 exit 1

Dependencies:
    pip install pyyaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPICS_DIR = REPO_ROOT / "topics"
INDEX_DIR = REPO_ROOT / "_index"

TOPIC_LABELS = {
    "llm": ("LLM", "大型語言模型:架構、預訓練、微調、對齊、推理"),
    "vlm": ("VLM", "視覺-語言模型、多模態理解與生成"),
    "contrastive-learning": ("Contrastive Learning", "對比學習、自監督表徵學習"),
    "rl": ("RL", "強化學習,包含 RLHF 的 RL 部分"),
    "agentic-ai": ("Agentic AI", "Agent 系統、工具使用、規劃、多 agent 協作"),
    "rag": ("RAG", "檢索增強生成、向量檢索與 LLM 結合"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PaperRef:
    """一篇被引用的論文(可能是種子,可能是 dependency)。"""

    title: str
    arxiv_id: str | None
    url: str | None
    relation: str | None = None

    @property
    def key(self) -> str:
        """論文唯一鍵——優先用 arxiv_id,沒有就用 title。"""
        return f"arxiv:{self.arxiv_id}" if self.arxiv_id else f"title:{self.title}"


@dataclass
class Article:
    """一篇解讀文章對應一個 meta.yaml。"""

    path: Path                # 相對於 repo root 的路徑
    topic: str
    paper_date: str           # YYYY-MM
    article_date: str         # YYYY-MM-DD
    title: str
    seed: PaperRef
    deps: list[PaperRef] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    knowledge_points: list[str] = field(default_factory=list)

    @property
    def article_dir_rel(self) -> str:
        """從 _index/ 出發的相對路徑。"""
        return f"../{self.path.relative_to(REPO_ROOT).as_posix()}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_articles() -> list[Article]:
    """掃描 topics/ 下所有 meta.yaml,解析成 Article 物件。"""
    articles: list[Article] = []

    if not TOPICS_DIR.exists():
        return articles

    for meta_path in TOPICS_DIR.glob("*/*/meta.yaml"):
        try:
            article = _parse_meta(meta_path)
            articles.append(article)
        except Exception as e:
            print(f"[WARN] 跳過 {meta_path}: {e}", file=sys.stderr)

    # 依論文發表時間反序(新到舊)
    articles.sort(key=lambda a: a.paper_date, reverse=True)
    return articles


def _parse_meta(meta_path: Path) -> Article:
    with open(meta_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    required = ["title", "topic", "paper_date", "article_date", "seed_paper"]
    for key in required:
        if key not in data:
            raise ValueError(f"缺少必填欄位: {key}")

    seed_data = data["seed_paper"]
    seed = PaperRef(
        title=seed_data["title"],
        arxiv_id=seed_data.get("arxiv_id"),
        url=seed_data.get("url"),
    )

    deps = [
        PaperRef(
            title=d["title"],
            arxiv_id=d.get("arxiv_id"),
            url=d.get("url"),
            relation=d.get("relation"),
        )
        for d in data.get("dependency_papers", []) or []
    ]

    return Article(
        path=meta_path.parent,
        topic=data["topic"],
        paper_date=str(data["paper_date"]),
        article_date=str(data["article_date"]),
        title=data["title"],
        seed=seed,
        deps=deps,
        tags=list(data.get("tags", []) or []),
        knowledge_points=list(data.get("knowledge_points", []) or []),
    )


# ---------------------------------------------------------------------------
# Render: by-date
# ---------------------------------------------------------------------------


def render_by_date(articles: list[Article]) -> str:
    """依論文發表時間,年/月分組。"""
    head = _index_header("By Date", "依**論文發表時間**(arxiv v1 發表年月)反序排列。")

    if not articles:
        body = "*尚無文章。第一篇文章發布後,此處會自動填入內容。*\n"
    else:
        grouped: dict[str, dict[str, list[Article]]] = defaultdict(lambda: defaultdict(list))
        for a in articles:
            year, month = a.paper_date.split("-")[:2]
            grouped[year][f"{year}-{month}"].append(a)

        parts: list[str] = []
        for year in sorted(grouped.keys(), reverse=True):
            parts.append(f"## {year}\n")
            for ym in sorted(grouped[year].keys(), reverse=True):
                parts.append(f"### {ym}\n")
                for a in grouped[year][ym]:
                    parts.append(_article_bullet(a))
                parts.append("")
        body = "\n".join(parts)

    stats = _stats_block([
        ("總文章數", len(articles)),
        ("最早論文", articles[-1].paper_date if articles else "暫無"),
        ("最新論文", articles[0].paper_date if articles else "暫無"),
    ])

    return f"{head}\n\n<!-- BEGIN: auto-generated content -->\n\n{body}\n<!-- END: auto-generated content -->\n\n---\n\n## 統計\n\n{stats}\n"


# ---------------------------------------------------------------------------
# Render: by-topic
# ---------------------------------------------------------------------------


def render_by_topic(articles: list[Article]) -> str:
    head = _index_header("By Topic", "依**主題分類**列出所有文章,每個主題內依論文發表時間反序。")

    by_topic: dict[str, list[Article]] = defaultdict(list)
    for a in articles:
        by_topic[a.topic].append(a)

    parts: list[str] = []
    all_topics = list(TOPIC_LABELS.keys()) + [
        t for t in by_topic if t not in TOPIC_LABELS
    ]
    seen = set()
    for topic in all_topics:
        if topic in seen:
            continue
        seen.add(topic)
        label, desc = TOPIC_LABELS.get(topic, (topic.title(), ""))
        parts.append(f"## {label}\n")
        if desc:
            parts.append(f"> {desc}\n")
        topic_articles = by_topic.get(topic, [])
        if not topic_articles:
            parts.append("*尚無文章*\n")
        else:
            for a in topic_articles:
                parts.append(_article_bullet(a))
        parts.append("")

    body = "\n".join(parts)

    active_topics = sum(1 for t in by_topic.values() if t)
    stats = _stats_block([
        ("總文章數", len(articles)),
        ("活躍 topic 數", f"{active_topics} / {len(TOPIC_LABELS)}"),
    ])

    return f"{head}\n\n<!-- BEGIN: auto-generated content -->\n\n{body}\n<!-- END: auto-generated content -->\n\n---\n\n## 統計\n\n{stats}\n"


# ---------------------------------------------------------------------------
# Render: by-paper
# ---------------------------------------------------------------------------


def render_by_paper(articles: list[Article]) -> str:
    head = _index_header(
        "By Paper",
        "依**論文反查**——以論文為主鍵,列出該論文被哪些文章涵蓋(作為種子論文或 dependency paper)。",
    )

    # 反向索引: paper_key -> {"seed": [articles], "dep": [(article, relation)]}
    inv: dict[str, dict[str, Any]] = {}

    for a in articles:
        key = a.seed.key
        inv.setdefault(key, {"paper": a.seed, "seed": [], "dep": []})
        inv[key]["seed"].append(a)

    for a in articles:
        for dep in a.deps:
            inv.setdefault(dep.key, {"paper": dep, "seed": [], "dep": []})
            inv[dep.key]["dep"].append((a, dep.relation))

    if not inv:
        body = "*尚無文章。第一篇文章發布後,此處會自動填入內容。*\n"
    else:
        # 按 arxiv_id 排序,沒有 arxiv_id 的排最後按 title
        def sort_key(item):
            paper = item[1]["paper"]
            if paper.arxiv_id:
                return (0, paper.arxiv_id)
            return (1, paper.title.lower())

        parts: list[str] = []
        for key, entry in sorted(inv.items(), key=sort_key):
            paper: PaperRef = entry["paper"]
            heading = f"arxiv:{paper.arxiv_id}" if paper.arxiv_id else paper.title
            parts.append(f"## {heading} — {paper.title}\n")

            parts.append("- **作為種子論文**:")
            seeds = entry["seed"]
            if seeds:
                for a in seeds:
                    parts.append(
                        f"  - [{a.title}]({a.article_dir_rel}) ({a.article_date} 撰寫)"
                    )
            else:
                parts.append("  - (無)")

            parts.append("- **作為 dependency**:")
            deps = entry["dep"]
            if deps:
                for a, relation in deps:
                    rel = f" — {relation}" if relation else ""
                    parts.append(f"  - [{a.title}]({a.article_dir_rel}){rel}")
            else:
                parts.append("  - (無)")

            parts.append("")
        body = "\n".join(parts)

    # 統計
    most_cited = ""
    if inv:
        ranked = sorted(
            inv.items(),
            key=lambda x: len(x[1]["seed"]) + len(x[1]["dep"]),
            reverse=True,
        )
        top = ranked[0]
        most_cited = (
            f"{top[1]['paper'].title}({len(top[1]['seed']) + len(top[1]['dep'])} 次)"
        )

    stats = _stats_block([
        ("涵蓋論文數", len(inv)),
        ("出現次數最多的論文", most_cited or "暫無"),
    ])

    return f"{head}\n\n<!-- BEGIN: auto-generated content -->\n\n{body}\n<!-- END: auto-generated content -->\n\n---\n\n## 統計\n\n{stats}\n"


# ---------------------------------------------------------------------------
# Render: reading-map
# ---------------------------------------------------------------------------


def render_reading_map(articles: list[Article]) -> str:
    head = _index_header(
        "Reading Map",
        "論文之間的**依賴關係圖**,以 Mermaid 呈現。藍色為有解讀文章的論文,灰色為僅作為 dependency 出現的論文。",
    )

    nodes: dict[str, PaperRef] = {}
    seed_keys: set[str] = set()
    edges: list[tuple[str, str]] = []

    for a in articles:
        nodes[a.seed.key] = a.seed
        seed_keys.add(a.seed.key)
        for dep in a.deps:
            nodes[dep.key] = dep
            # dep -> seed (因為 seed 依賴 dep)
            edges.append((dep.key, a.seed.key))

    if not nodes:
        body = "*尚無文章。第一篇文章發布後,此處會自動生成依賴圖。*\n"
    else:
        lines = ["```mermaid", "graph TD"]
        node_ids: dict[str, str] = {}
        for i, (key, paper) in enumerate(nodes.items()):
            nid = f"n{i}"
            node_ids[key] = nid
            label = _node_label(paper)
            lines.append(f'    {nid}["{label}"]')

        lines.append("")
        for src, dst in edges:
            lines.append(f"    {node_ids[src]} --> {node_ids[dst]}")

        lines.append("")
        lines.append("    classDef seed fill:#e1f5ff,stroke:#0066cc")
        lines.append("    classDef dep fill:#f0f0f0,stroke:#666")
        seed_nodes = [node_ids[k] for k in seed_keys if k in node_ids]
        dep_nodes = [node_ids[k] for k in nodes if k not in seed_keys]
        if seed_nodes:
            lines.append(f"    class {','.join(seed_nodes)} seed")
        if dep_nodes:
            lines.append(f"    class {','.join(dep_nodes)} dep")
        lines.append("```")
        body = "\n".join(lines) + "\n"

    stats = _stats_block([
        ("節點數", len(nodes)),
        ("邊數", len(edges)),
    ])

    return f"{head}\n\n<!-- BEGIN: auto-generated content -->\n\n{body}\n<!-- END: auto-generated content -->\n\n---\n\n## 統計\n\n{stats}\n"


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _index_header(name: str, desc: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Index: {name}\n\n"
        f"> 🤖 **此檔案由 `scripts/generate_index.py` 自動生成,請勿手動編輯。**\n"
        f"> 最後更新:{ts}\n\n"
        f"{desc}\n\n---"
    )


def _article_bullet(a: Article) -> str:
    tag_str = " · ".join(f"`{t}`" for t in a.tags[:4]) if a.tags else ""
    tag_line = f"\n  - Tags: {tag_str}" if tag_str else ""
    arxiv_line = ""
    if a.seed.arxiv_id and a.seed.url:
        arxiv_line = f"\n  - 種子論文:[{a.seed.title} (arxiv:{a.seed.arxiv_id})]({a.seed.url})"
    return f"- **[{a.title}]({a.article_dir_rel})** ({a.paper_date}){tag_line}{arxiv_line}"


def _node_label(paper: PaperRef) -> str:
    """Mermaid 節點標籤——簡短,加 arxiv id 作後綴。"""
    title = paper.title.split(":")[0].strip()
    if len(title) > 30:
        title = title[:27] + "..."
    suffix = f"<br/>{paper.arxiv_id}" if paper.arxiv_id else ""
    return f"{title}{suffix}"


def _stats_block(items: list[tuple[str, Any]]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in items)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


INDEX_FILES = {
    "by-date.md": render_by_date,
    "by-topic.md": render_by_topic,
    "by-paper.md": render_by_paper,
    "reading-map.md": render_reading_map,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Paper Lens index files.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print to stdout, don't write files."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if index files are outdated (for CI).",
    )
    args = parser.parse_args()

    articles = load_articles()
    print(f"[INFO] 載入 {len(articles)} 篇文章", file=sys.stderr)

    outdated: list[str] = []

    for filename, renderer in INDEX_FILES.items():
        new_content = renderer(articles)
        target = INDEX_DIR / filename

        if args.dry_run:
            print(f"\n{'=' * 60}\n# {filename}\n{'=' * 60}\n")
            print(new_content)
            continue

        if args.check:
            if not target.exists() or _strip_timestamp(target.read_text(encoding="utf-8")) != _strip_timestamp(new_content):
                outdated.append(filename)
            continue

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        print(f"[OK] 寫入 {target.relative_to(REPO_ROOT)}", file=sys.stderr)

    if args.check and outdated:
        print(
            f"[ERROR] 以下索引檔案過期: {', '.join(outdated)}\n"
            f"請執行 `python scripts/generate_index.py` 更新後重新提交。",
            file=sys.stderr,
        )
        return 1

    return 0


def _strip_timestamp(content: str) -> str:
    """比對內容時忽略時間戳行,避免時間戳差異造成誤判。"""
    return "\n".join(
        line for line in content.splitlines() if not line.startswith("> 最後更新:")
    )


if __name__ == "__main__":
    sys.exit(main())
