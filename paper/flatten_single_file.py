"""Flatten coding_agents.tex into one self-contained coding_agents_full.tex (figures as embedded PGF).

    python3 paper/flatten_single_file.py paper paper/figures
"""

import re, sys
from pathlib import Path
P = Path(sys.argv[1]); PGF = Path(sys.argv[2])
def inline(text):
    def rep(m):
        f = m.group(1); f = f if f.endswith('.tex') else f + '.tex'
        return f'% ---- begin {f} ----\n' + inline((P / f).read_text().rstrip('\n')) + f'\n% ---- end {f} ----'
    return re.sub(r'\\input\{([^}]+)\}', rep, text)
s = inline((P / 'coding_agents.tex').read_text())
used = []
def fig(m):
    name = Path(m.group(2)).stem; used.append(name)
    return '\\resizebox{%s}{!}{\\input{%s.pgf}}' % (m.group(1), name)
s = re.sub(r'\\includegraphics\[width=([^\]]+)\]\{([^}]+)\}', fig, s)
bbl = (P / 'coding_agents.bbl').read_text().rstrip('\n')
s = re.sub(r'\\bibliographystyle\{plainnat\}\n\\bibliography\{references\}', lambda m: '% ---- bibliography (from references.bib, plainnat) ----\n' + bbl, s)
assert '\\bibliography{' not in s and '\\includegraphics' not in s
s = s.replace('\\usepackage{pifont}', '\\usepackage{pifont}\n\\providecommand{\\mathdefault}[1]{#1}% used by the embedded matplotlib figures', 1)
blocks = ''.join('\\begin{filecontents*}[overwrite]{%s.pgf}\n' % n + (PGF / (n + '.pgf')).read_text().rstrip('\n') + '\n\\end{filecontents*}\n' for n in used)
head = ('% Can Decision Models Replace Tool Calling for Coding Agents?\n'
        '% Self-contained source: preamble, tables, bibliography and figures are all in this file.\n'
        '% The two plots are embedded as PGF code (matplotlib export); LaTeX writes them to fig_*.pgf when compiling.\n'
        '% Build: pdflatex coding_agents_full.tex  (run it twice for cross-references)\n')
s = s.replace('% Can Decision Models Replace Tool Calling for Coding Agents?\n', head + '%\n% ===== embedded figures =====\n' + blocks + '% ===== end of embedded figures =====\n', 1)
(P / 'coding_agents_full.tex').write_text(s)
print(used, len(s.splitlines()), 'lines')
