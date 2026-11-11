"""Typeset the manuscript with XeLaTeX, retaining equations and vector figures."""
import re
import os
from pathlib import Path
from urllib.parse import quote, unquote

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
    return "".join(mapping.get(character, character) for character in text)


def main():
    source = (PAPER / "共享LoRA中的旧事实干扰.md").read_text(encoding="utf8")
    equations = {}
    def protect(match):
        key = f"MATHPLACEHOLDER{len(equations)}END"
        raw = match.group(0)
        equations[key] = r"\[" + raw[2:-2] + r"\]" if raw.startswith("$$") else raw
        return key
    source = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$\n]+\$", protect, source)
    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(source)

    def inline(children):
        output = []
        for token in children or []:
            if token.type == "text":
                output.append(escape(token.content))
            elif token.type in ("softbreak", "hardbreak"):
                output.append(" ")
            elif token.type == "strong_open":
                output.append(r"\textbf{")
            elif token.type == "em_open":
                output.append(r"\emph{")
            elif token.type in ("strong_close", "em_close", "link_close"):
                output.append("}")
            elif token.type == "code_inline":
                output.append(r"\texttt{" + escape(token.content) + "}")
            elif token.type == "link_open":
                url = quote(token.attrGet("href"), safe=":/?&=#%").replace("%", r"\%").replace("#",r"\#")
                output.append(r"\href{" + url + "}{")
            elif token.type == "image":
                src = unquote(token.attrGet("src"))
                if "/blob/main/" in src:
                    path = ROOT / src.split("/blob/main/")[1]
                elif "/raw/main/" in src:
                    path = ROOT / src.split("/raw/main/")[1]
                elif "raw.githubusercontent.com/" in src and "/main/" in src:
                    path = ROOT / src.split("/main/",1)[1]
                else:
                    path = (PAPER / src).resolve()
                if path.suffix == ".png" and path.with_suffix(".pdf").exists():
                    path = path.with_suffix(".pdf")
                path = Path(os.path.relpath(path,PAPER))
                output.append(r"\includegraphics[width=\linewidth,height=.65\textheight,"
                              r"keepaspectratio]{" + path.as_posix() + r"}")
        return "".join(output)

    body = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (token.type == "paragraph_open" and i+1 < len(tokens)
                and tokens[i+1].type == "inline"
                and any(t.type == "image" for t in tokens[i+1].children or [])):
            body.append(r"\begin{figure}[H]\centering" + "\n" + inline(tokens[i+1].children))
            i += 3
            if (i+1 < len(tokens) and tokens[i].type == "paragraph_open"
                    and tokens[i+1].type == "inline" and tokens[i+1].content.startswith("图")):
                body.append(r"\caption*{" + inline(tokens[i+1].children) + "}")
                i += 3
            body.append(r"\end{figure}")
            continue
        if token.type == "heading_open":
            title = inline(tokens[i+1].children)
            level = int(token.tag[1])
            if level == 1:
                body.append(r"\title{" + title + "}\n" + r"\maketitle")
            else:
                command = "section" if level == 2 else "subsection" if level == 3 else "subsubsection"
                body.append("\\" + command + "*{" + title + "}")
            i += 3
            continue
        if token.type == "table_open":
            rows, row = [], []
            i += 1
            while tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    row = []
                elif tokens[i].type == "inline":
                    row.append(inline(tokens[i].children))
                elif tokens[i].type == "tr_close":
                    rows.append(row)
                i += 1
            n = len(rows[0])
            column = r">{\raggedright\arraybackslash}p{\dimexpr\linewidth/" + str(n) + r"-2\tabcolsep\relax}"
            body.append(r"\begingroup\small\setlength{\tabcolsep}{3pt}" +
                        r"\begin{longtable}{@{}" + column*n + r"@{}}\toprule")
            header = " & ".join(r"\textbf{" + cell + "}" for cell in rows[0]) + r"\\\midrule"
            body.append(header + r"\endfirsthead" + header + r"\endhead")
            body.extend(" & ".join(row) + r"\\" for row in rows[1:])
            body.append(r"\bottomrule\end{longtable}\endgroup")
        elif token.type == "inline":
            body.append(inline(token.children))
        elif token.type == "paragraph_close":
            body.append("\n\n")
        elif token.type == "bullet_list_open":
            body.append(r"\begin{itemize}")
        elif token.type == "bullet_list_close":
            body.append(r"\end{itemize}")
        elif token.type == "ordered_list_open":
            body.append(r"\begin{enumerate}")
        elif token.type == "ordered_list_close":
            body.append(r"\end{enumerate}")
        elif token.type == "list_item_open":
            body.append(r"\item")
        elif token.type == "fence":
            body.append(r"\begin{Verbatim}[breaklines=true]" + "\n" + token.content + r"\end{Verbatim}")
        i += 1
    content = "\n".join(body)
    for key, value in equations.items():
        content = content.replace(key,value)
    preamble = r"""\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,hyperref,fvextra,float,caption}
\captionsetup{font=small,justification=raggedright,singlelinecheck=false}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{2em}
\makeatletter
\renewcommand{\maketitle}{\begin{center}{\LARGE\bfseries\@title\par}\end{center}\vspace{.4em}}
\makeatother
\author{}
\date{}
\begin{document}
"""
    target = PAPER / "manuscript.tex"
    target.write_text(preamble + content + "\n" + r"\end{document}",encoding="utf8")
    print(target)


if __name__ == "__main__":
    main()
