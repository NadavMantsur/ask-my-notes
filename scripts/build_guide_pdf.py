#!/usr/bin/env python3
"""Convert docs/RAG-GUIDE.md into a printable PDF.

This is not part of the RAG pipeline. It exists so you can read the same
explanation offline without installing a Markdown previewer.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "docs" / "RAG-GUIDE.md"
OUTPUT = PROJECT_ROOT / "docs" / "ask-my-notes-rag-guide.pdf"


class GuidePDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "ask-my-notes  |  from-scratch RAG guide", align="L")
        self.ln(12)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def latin(text: str) -> str:
    """Helvetica is Latin-1; fold a few Unicode punctuation marks."""
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
        "**": "",
        "`": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.encode("latin-1", "replace").decode("latin-1")


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    pdf = GuidePDF(format="Letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.add_page()
    pdf.set_y(80)
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 12, "ask-my-notes", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(0, 8, "A from-scratch RAG tutorial", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(
        0,
        7,
        "Architecture, file structure, tools, and the six pipeline phases.\n"
        "Read this even if you have not opened src/ yet.",
        align="C",
    )

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Contents")
    pdf.ln(12)
    headings = [
        latin(line[3:].strip())
        for line in markdown.splitlines()
        if line.startswith("## ")
    ]
    pdf.set_font("Helvetica", "", 11)
    for i, title in enumerate(headings, start=1):
        pdf.cell(0, 7, f"{i}.  {title}")
        pdf.ln(7)

    pdf.add_page()
    in_code = False
    skipped_title = False
    paragraph: list[str] = []

    def write_cell(font: str, style: str, size: int, height: float, text: str, fill: bool = False) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font(font, style, size)
        usable = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.multi_cell(usable, height, text, fill=fill)

    def flush_paragraph() -> None:
        if not paragraph:
            return
        write_cell("Helvetica", "", 11, 6, latin(" ".join(paragraph)))
        pdf.ln(2)
        paragraph.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                in_code = False
                pdf.ln(3)
            else:
                in_code = True
                pdf.set_fill_color(245, 245, 245)
            continue
        if in_code:
            write_cell("Courier", "", 8, 4.5, latin(line) or " ", fill=True)
            continue
        if not line.strip():
            flush_paragraph()
            continue
        if line.strip() == "---":
            flush_paragraph()
            continue
        if line.startswith("# ") and not skipped_title:
            skipped_title = True
            continue
        if line.startswith("# "):
            flush_paragraph()
            write_cell("Helvetica", "B", 18, 9, latin(line[2:]))
            pdf.ln(3)
        elif line.startswith("## "):
            flush_paragraph()
            pdf.ln(3)
            write_cell("Helvetica", "B", 14, 8, latin(line[3:]))
            pdf.ln(2)
        elif line.startswith("### "):
            flush_paragraph()
            write_cell("Helvetica", "B", 12, 7, latin(line[4:]))
            pdf.ln(1)
        elif line.startswith("|"):
            flush_paragraph()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            write_cell("Helvetica", "", 9, 5, latin(" | ".join(cells)))
        elif line.startswith("- "):
            flush_paragraph()
            write_cell("Helvetica", "", 11, 6, f"  - {latin(line[2:])}")
        elif re.match(r"^\d+\.\s", line):
            flush_paragraph()
            write_cell("Helvetica", "", 11, 6, f"  {latin(line)}")
        else:
            paragraph.append(line.strip())
    flush_paragraph()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
