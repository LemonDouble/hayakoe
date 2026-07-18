"""HTML 리포트 공통 디자인 토큰 · 기본 스타일.

benchmark / report HTML 리포트가 같은 룩을 공유하도록 base CSS 를 한 곳에
둔다. 각 리포트는 ``BASE_CSS + 전용 규칙`` 으로 조합해 쓰고, 컨테이너 폭
등이 다르면 전용 규칙에서 덮어쓴다.
"""

BASE_CSS = """\
@import url('https://cdn.jsdelivr.net/npm/galmuri/dist/galmuri.css');
@import url('https://cdnjs.cloudflare.com/ajax/libs/pretendard/1.3.9/static/pretendard.min.css');
:root{
  --color-primary:#F0B90B;--color-secondary:#CD6B5E;
  --color-bg-dark:#12100E;--color-surface:#1C1816;--color-surface-hover:#231E1B;
  --color-border:#2E2723;--color-border-hover:#3D3530;
  --color-text-primary:#F5F0EB;--color-text-secondary:#A89E95;--color-text-muted:#5C524A;
  --color-primary-dim:rgba(240,185,11,0.12);--color-primary-dim-border:rgba(240,185,11,0.25);
  --color-good:#4ADE80;--color-ok:#F0B90B;--color-slow:#CD6B5E;
  --font-heading:'Galmuri11',monospace;
  --font-body:'Pretendard',-apple-system,BlinkMacSystemFont,sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--color-bg-dark);color:var(--color-text-secondary);font-family:var(--font-body);line-height:1.6;font-size:15px}
.c{max-width:1000px;margin:0 auto;padding:32px 24px}
header{margin-bottom:32px;padding-bottom:24px;border-bottom:1px solid var(--color-border)}
h1{font-family:var(--font-heading);font-size:28px;font-weight:700;line-height:1.3;color:var(--color-text-primary);margin-bottom:8px}
.sub{color:var(--color-text-muted);font-size:13px}
h2{font-family:var(--font-heading);font-size:20px;font-weight:700;line-height:1.4;color:var(--color-text-primary);margin-bottom:16px}
section{margin-bottom:40px}
.tw{overflow-x:auto;border-radius:12px;border:1px solid var(--color-border);background:var(--color-surface)}
table{width:100%;border-collapse:collapse}
thead{background:var(--color-bg-dark)}
th{padding:12px 16px;font-family:var(--font-body);font-size:13px;font-weight:600;color:var(--color-text-secondary);text-align:center;border-bottom:1px solid var(--color-border);white-space:nowrap;position:sticky;top:0;z-index:1;background:var(--color-bg-dark)}
td{padding:12px 16px;font-size:13px;color:var(--color-text-primary);border-bottom:1px solid var(--color-border);text-align:center;vertical-align:middle}
tbody tr:hover td{background:var(--color-surface-hover)}
tbody tr:last-child td{border-bottom:none}
.ft{text-align:center;color:var(--color-text-muted);font-size:12px;margin-top:32px;padding-top:16px;border-top:1px solid var(--color-border)}
"""
