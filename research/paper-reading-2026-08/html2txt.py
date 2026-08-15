"""Convert arXiv HTML paper to readable text with structure preserved."""
import re, sys
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = 0
        self.in_math = 0
        self.math_buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip += 1
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'tr', 'figcaption', 'caption', 'table', 'figure'):
            self.out.append('\n')
        if tag in ('h1',): self.out.append('\n# ')
        if tag in ('h2',): self.out.append('\n## ')
        if tag in ('h3',): self.out.append('\n### ')
        if tag in ('h4',): self.out.append('\n#### ')
        if tag == 'li': self.out.append('- ')
        if tag == 'td': self.out.append(' | ')
        if tag == 'tr': self.out.append('\n')
        if tag == 'math': self.in_math += 1
        if tag == 'br': self.out.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = max(0, self.skip - 1)
        if tag == 'math':
            self.in_math = max(0, self.in_math - 1)
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'tr', 'figcaption', 'table', 'figure'):
            self.out.append('\n')

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_math:
            # keep math as raw text-ish (annotation-xml or alttext lost; just keep chars)
            self.out.append(data)
        else:
            self.out.append(data)

def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    with open(src, encoding='utf-8', errors='replace') as f:
        html = f.read()
    # drop SVG junk to reduce noise
    html = re.sub(r'<svg.*?</svg>', ' ', html, flags=re.S)
    p = TextExtractor()
    p.feed(html)
    text = ''.join(p.out)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"OK: {len(text)} chars -> {dst}")

if __name__ == '__main__':
    main()
