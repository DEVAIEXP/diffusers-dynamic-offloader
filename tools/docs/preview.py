"""Expose public documentation without serving the checkout root."""

from pathlib import Path
import re

from mkdocs.structure.files import File, Files


def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    docs = root / "docs"
    public = Files([])
    for file in files:
        if Path(file.src_dir).resolve() != docs.resolve():
            public.append(file)
            continue
        source = Path(file.abs_src_path).resolve(strict=True)
        if not source.is_relative_to(docs):
            raise ValueError(f"Documentation source escapes docs/: {source}")
        public.append(File.generated(config, f"docs/{file.src_uri}", abs_src_path=str(source)))
    for relative in ("README.md", "AGENTS.md"):
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError(f"Public documentation source escapes repository: {relative}")
        public.append(File.generated(config, relative, abs_src_path=str(source)))
    return public


def on_page_content(html, page, config, files):
    """Keep wide tables scrollable without wrapping their column headings."""
    return re.sub(
        r"(<table\b[^>]*>.*?</table>)",
        r'<div class="table-scroll" role="region" aria-label="Scrollable table" tabindex="0">\1</div>',
        html,
        flags=re.DOTALL,
    )
