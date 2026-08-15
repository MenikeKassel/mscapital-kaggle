#!/usr/bin/env python3
"""Verify arXiv IDs via abs pages (server-rendered HTML). Saves title/abstract JSON."""
import json, os, re, sys, time, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abs_verified.json')

def fetch_abs(arxiv_id, retries=4):
    url = f'https://arxiv.org/abs/{arxiv_id}'
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) research/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f'    [retry {attempt+1}] {e} wait {wait}s', flush=True)
            time.sleep(wait)
    return None

def parse_abs(html):
    if not html:
        return None
    m = re.search(r'<title>\[([0-9.]+)\]\s*(.*?)</title>', html, re.S)
    title = re.sub(r'\s+', ' ', m.group(2)).strip() if m else '?'
    m2 = re.search(r'<blockquote class="abstract[^"]*">\s*<span[^>]*>(.*?)</span>', html, re.S)
    abstract = re.sub(r'\s+', ' ', m2.group(1)).strip() if m2 else '?'
    m3 = re.search(r'<div class="submission-history">.*?<span class="submission-history__item">(.*?)</span>', html, re.S)
    date = re.sub(r'\s+', ' ', m3.group(1)).strip() if m3 else '?'
    m4 = re.search(r'<meta name="citation_doi" content="([^"]+)"', html)
    doi = m4.group(1) if m4 else ''
    m5 = re.search(r'<div class="authors">(.*?)</div>', html, re.S)
    authors = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', m5.group(1))).strip() if m5 else '?'
    return {'title': title, 'abstract': abstract, 'submission_history': date,
            'doi': doi, 'authors': authors}

def main():
    ids = [l.strip() for l in open('idlist.txt') if l.strip()]
    results = {}
    for i, aid in enumerate(ids):
        print(f'== {aid}', flush=True)
        parsed = parse_abs(fetch_abs(aid))
        if parsed:
            results[aid] = parsed
            print(f"   ✓ {parsed['title'][:100]}")
            print(f"   ({parsed['submission_history'][:60]}) DOI:{parsed['doi'] or '-'}")
        else:
            print('   ✗ NOT FOUND / ERROR')
        time.sleep(3)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nSaved {len(results)}/{len(ids)} -> {OUT}')

if __name__ == '__main__':
    main()
