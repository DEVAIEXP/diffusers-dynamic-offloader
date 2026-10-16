/* Render Mermaid fences while retaining readable source if rendering fails. */
(async () => {
  const render = async () => {
    const blocks = [...document.querySelectorAll("pre > code.language-mermaid")];
    if (!blocks.length) return;
    const replacements = [];
    try {
      const { default: mermaid } = await import(
        "https://cdn.jsdelivr.net/npm/mermaid@12.2.0/dist/mermaid.esm.min.mjs"
      );
      mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
      for (const code of blocks) {
        const original = code.parentElement;
        const diagram = document.createElement("pre");
        diagram.className = "mermaid";
        diagram.textContent = code.textContent;
        original.replaceWith(diagram);
        replacements.push({ original, diagram });
      }
      await mermaid.run({ nodes: replacements.map(({ diagram }) => diagram) });
    } catch (error) {
      for (const { original, diagram } of replacements) diagram.replaceWith(original);
      console.warn("Mermaid rendering unavailable; diagram source remains visible.", error);
    }
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render, { once: true });
  } else {
    await render();
  }
})();
