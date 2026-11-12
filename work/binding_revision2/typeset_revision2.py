"""Render a separate working manuscript; preserve the historical TeX and PDF."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"work/binding_review"))
import typeset

plain_escape=typeset.escape
symbols={"Σ":r"\(\Sigma\)","Δ":r"\(\Delta\)","β":r"\(\beta\)","ε":r"\(\epsilon\)",
         "−":r"\(-\)","≤":r"\(\leq\)","≥":r"\(\geq\)","×":r"\(\times\)","∈":r"\(\in\)"}
typeset.escape=lambda value:"".join(symbols[c] if c in symbols else plain_escape(c) for c in value)

if __name__=="__main__":
    typeset.main("共享LoRA中的条件绑定干扰_第二轮工作稿.md","manuscript_revision2.tex",blockquote_environment="findingbox")
