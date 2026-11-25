"""Word count for the seminar paper, checked against the course limits.

    python wordcount.py

Reads main.tex, follows its \\input{...} files and counts words per file in
the same spirit as texcount (the tool behind Overleaf's word count):

  * text      running text, including words inside \\textit{} etc.
  * headings  section and subsection titles
  * captions  figure and table captions (listed separately)

Not counted: comments, math, citations/references/labels, table cells,
the bibliography and \\fillin{...} placeholders (reported as "todo").
The appendix (Critical Observations) and the abstract are reported
separately. Whether headings, captions and the abstract count towards the
1400 words is for the course to decide; all variants are printed.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BODY_TARGET = (1330, 1470)       # 1400 words +/- 5 %
APPENDIX_TARGET = (285, 315)     # 300 words +/- 5 %
WORD = re.compile(r"[A-Za-z0-9\u00C0-\u024F]+(?:[-'\u2019][A-Za-z0-9\u00C0-\u024F]+)*")


def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def take_group(tex, start):
    """Return (content, end) of the {...} group starting at tex[start] == '{'."""
    depth = 0
    for i in range(start, len(tex)):
        if tex[i] == "{" and tex[i - 1] != "\\":
            depth += 1
        elif tex[i] == "}" and tex[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return tex[start + 1:i], i + 1
    return tex[start + 1:], len(tex)


def extract(tex, command):
    """Remove every \\command{...} (also \\command*{...}); return (tex, args)."""
    pattern = re.compile(r"\\" + command + r"\*?\s*(\[[^\]]*\])?\s*(?=\{)")
    args, out, pos = [], [], 0
    while (m := pattern.search(tex, pos)):
        content, end = take_group(tex, m.end())
        args.append(content)
        out.append(tex[pos:m.start()])
        pos = end
    out.append(tex[pos:])
    return "".join(out), args


def remove_env(tex, env):
    return re.sub(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env)
                  + r"\}", " ", tex, flags=re.S)


def plain_words(tex):
    tex = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|(?<!\\)\$.*?(?<!\\)\$", " ", tex, flags=re.S)
    tex = re.sub(r"\\(begin|end)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)   # commands, keep {args}
    tex = re.sub(r"\\.", " ", tex)                           # \, \% ...
    tex = tex.replace("~", " ").replace("{", " ").replace("}", " ")
    return len(WORD.findall(tex))


def count(tex):
    tex = strip_comments(tex)
    tex, todo = extract(tex, "fillin")
    tex, captions = extract(tex, "caption")
    for env in ("figure", "figure*", "table", "table*", "tabular",
                "equation", "equation*", "align", "align*", "IEEEeqnarray"):
        tex = remove_env(tex, env)
    tex, headings = extract(tex, "(?:sub)*section")
    for cmd in ("cite", "ref", "eqref", "label", "includegraphics", "url",
                "bibliography", "bibliographystyle"):
        tex, _ = extract(tex, cmd)
    return {
        "text": plain_words(tex),
        "headings": sum(plain_words(h) for h in headings),
        "captions": sum(plain_words(c) for c in captions),
        "todo": len(todo),
    }


def inputs(main):
    """Files \\input in main.tex, in order."""
    tex = strip_comments(main.read_text(encoding="utf-8"))
    for name in re.findall(r"\\input\{([^}]*)\}", tex):
        path = HERE / name
        yield path if path.suffix == ".tex" else path.with_suffix(".tex")


def status(n, lo, hi):
    if n < lo:
        return f"{lo - n} words short (min {lo})"
    if n > hi:
        return f"{n - hi} words over (max {hi})"
    return f"OK ({lo}-{hi})"


def main():
    body = {"text": 0, "headings": 0, "captions": 0, "todo": 0}
    abstract = appendix = None
    print(f"{'file':48s} {'text':>6} {'heads':>6} {'capts':>6} {'todo':>5}")
    for path in inputs(HERE / "main.tex"):
        c = count(path.read_text(encoding="utf-8"))
        print(f"{path.relative_to(HERE).as_posix():48s} {c['text']:6d} "
              f"{c['headings']:6d} {c['captions']:6d} {c['todo']:5d}")
        if "abstract" in path.name:
            abstract = c
        elif "appendix" in path.name:
            appendix = c
        else:
            for k in body:
                body[k] += c[k]

    text_heads = body["text"] + body["headings"]
    print()
    print(f"Body, text + headings:             {text_heads:5d}   "
          f"{status(text_heads, *BODY_TARGET)}")
    print(f"Body, text + headings + captions:  {text_heads + body['captions']:5d}   "
          f"{status(text_heads + body['captions'], *BODY_TARGET)}")
    if abstract:
        with_abs = text_heads + abstract["text"]
        print(f"Body, text + headings + abstract:  {with_abs:5d}   "
              f"{status(with_abs, *BODY_TARGET)}")
    if appendix:
        print(f"Appendix (text only):              {appendix['text']:5d}   "
              f"{status(appendix['text'], *APPENDIX_TARGET)}")
    todo = body["todo"] + sum(c["todo"] for c in (abstract, appendix) if c)
    if todo:
        print(f"\n{todo} \\fillin placeholders left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
