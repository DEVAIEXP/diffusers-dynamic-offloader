# Local documentation preview

The public site is <https://devaiexp.github.io/diffusers-dynamic-offloader/>.
The `docs-pages.yml` workflow builds with MkDocs 1.6.1 in strict mode and
publishes through GitHub Pages after documentation changes reach `main`.
Maintainers can also dispatch it manually. Only the generated static site is
uploaded; inference packages, submodules, local outputs and private records
are not needed. GitHub Pages must use GitHub Actions as its publishing source.

The commands below remain a local preview and do not publish anything.

From the DDO repository root, run:

```text
uvx --from mkdocs==1.6.1 mkdocs serve --config-file mkdocs.yml
```

Open <http://127.0.0.1:8001/>. DDO uses port 8001 so that the examples preview
can remain available on port 8000. Stop the server with Ctrl+C.
Use `--dev-addr 127.0.0.1:8002` if port 8001 is occupied.

The pinned MkDocs tool uses its own uv cache, not the inference environment.
Its first installation needs network access; add `--offline` to `uvx` when the
tool is already cached and network access is unavailable.
No CUDA, model loading or native build is involved.

The preview serves `docs/` plus the public README and agent guidance through
virtual files, preserving repository-relative links. It does not serve the
checkout root, environments, weights, builds or private records. Generated HTML
stays under ignored `.internal/docs-preview/site/`; nothing is published.

Browser translation can display another language without changing the English
sources. Translation may send page text to the provider; copy commands from
the original English version.

Mermaid diagrams render in the browser using a pinned Mermaid 12.2.0 module
from jsDelivr. Rendering requires access to that CDN; if it is unavailable,
the original diagram source remains visible. No inference dependency is added.

To check the documentation without starting a server:

```text
uvx --from mkdocs==1.6.1 mkdocs build --config-file mkdocs.yml --strict
```
