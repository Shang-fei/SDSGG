from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent

SLIDES = [
    ("标题页", "开放词汇场景图生成\n新颖谓词语义结构迁移"),
    ("汇报目录", "背景 / 动机 / 方法 / 实验 / 缺陷"),
    ("SGG是什么", "Image → Scene Graph\nSubject-Predicate-Object"),
    ("闭集到开放词汇", "Closed-set vocabulary\n→ Novel relations"),
    ("开放词汇关系更难", "Predicate depends on\nsubject-object interaction"),
    ("VLM/LLM驱动OV-SGG", "VLM alignment\nLLM enriched prompts"),
    ("动态Prompt现状", "Adaptive / Relation-aware\nPrompt"),
    ("现有方法不足", "Scene adaptation may not\npreserve semantic structure"),
    ("本文研究问题", "How to transfer predicate\nsemantic structure?"),
    ("核心思路", "SVD + Structure Loss\n+ SHIP Generation"),
    ("方法总览", "SDSGG feature → semantic\nstructure transfer"),
    ("SVD低秩语义", "Compact predicate\nsemantic basis"),
    ("结构保持对齐", "Visual structure ≈\nText semantic structure"),
    ("SHIP生成", "Generated features for\nnovel predicates"),
    ("消融设置", "SVD only / SVD+Structure\n/ Full Model"),
    ("实验设置", "VG PredCls\nR@20 / R@50 / R@100"),
    ("Base类结果", "Structure improves\nbase stability"),
    ("Novel类结果", "SHIP improves\nnovel recall"),
    ("Trade-off分析", "Base stability ↔\nNovel transfer"),
    ("缺陷与计划", "PredCls only / seeds /\nGQA or new split"),
]


THEMES = [
    {
        "name": "version_a_modern_blue",
        "bg": "#F7FAFC",
        "slide": "#FFFFFF",
        "ink": "#122033",
        "muted": "#5B677A",
        "primary": "#155EEF",
        "accent": "#0EA5E9",
        "line": "#D7DEE9",
        "soft": "#EAF3FF",
        "title": "版本A：蓝白极简学术风",
    },
    {
        "name": "version_b_dark_tech",
        "bg": "#0D1321",
        "slide": "#141C2F",
        "ink": "#EEF4FF",
        "muted": "#A8B3C7",
        "primary": "#7DD3FC",
        "accent": "#A78BFA",
        "line": "#2B3650",
        "soft": "#1F2A44",
        "title": "版本B：深色科技风",
    },
    {
        "name": "version_c_warm_paper",
        "bg": "#F6F2EC",
        "slide": "#FFFCF7",
        "ink": "#202124",
        "muted": "#706A61",
        "primary": "#9A5B2E",
        "accent": "#2F7D68",
        "line": "#DED5C8",
        "soft": "#EFE5D6",
        "title": "版本C：暖灰论文汇报风",
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=0)
    return ImageFont.load_default()


F_TITLE = font(44)
F_SUB = font(22)
F_SLIDE_TITLE = font(20)
F_SLIDE_BODY = font(15)
F_NUM = font(13)


def rounded(draw: ImageDraw.ImageDraw, box, r, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_network(draw, x, y, w, h, theme, idx):
    pts = [
        (x + w * 0.22, y + h * 0.25),
        (x + w * 0.52, y + h * 0.18),
        (x + w * 0.78, y + h * 0.35),
        (x + w * 0.35, y + h * 0.68),
        (x + w * 0.68, y + h * 0.74),
    ]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4), (1, 4)]:
        draw.line([pts[a], pts[b]], fill=theme["line"], width=2)
    for i, p in enumerate(pts):
        fill = theme["primary"] if i % 2 == idx % 2 else theme["accent"]
        draw.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=fill)


def draw_bars(draw, x, y, w, h, theme, idx):
    vals = [0.52, 0.68, 0.76, 0.61]
    gap = w / 6
    for i, v in enumerate(vals):
        bx = x + gap * (i + 1)
        bw = gap * 0.55
        by = y + h * (1 - v)
        color = theme["primary"] if i < 2 else theme["accent"]
        rounded(draw, (bx, by, bx + bw, y + h), 4, color)
    draw.line([x, y + h, x + w, y + h], fill=theme["line"], width=2)


def draw_table(draw, x, y, w, h, theme):
    rows, cols = 4, 4
    for r in range(rows + 1):
        yy = y + h * r / rows
        draw.line([x, yy, x + w, yy], fill=theme["line"], width=1)
    for c in range(cols + 1):
        xx = x + w * c / cols
        draw.line([xx, y, xx, y + h], fill=theme["line"], width=1)
    for c in range(1, cols):
        draw.ellipse([x + w*c/cols - 5, y + h*0.62 - 5, x + w*c/cols + 5, y + h*0.62 + 5], fill=theme["accent"])


def draw_tradeoff(draw, x, y, w, h, theme):
    draw.line([x + 15, y + h - 15, x + w - 10, y + h - 15], fill=theme["line"], width=2)
    draw.line([x + 15, y + h - 15, x + 15, y + 10], fill=theme["line"], width=2)
    pts = [(x + 35, y + h - 42), (x + w * 0.48, y + h * 0.45), (x + w - 35, y + 35)]
    draw.line(pts, fill=theme["primary"], width=3)
    for p in pts:
        draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill=theme["accent"])


def draw_icon(draw, x, y, w, h, theme, idx):
    kind = idx % 5
    if kind == 0:
        draw_network(draw, x, y, w, h, theme, idx)
    elif kind == 1:
        draw_bars(draw, x, y, w, h, theme, idx)
    elif kind == 2:
        draw_table(draw, x, y, w, h, theme)
    elif kind == 3:
        draw_tradeoff(draw, x, y, w, h, theme)
    else:
        for i in range(3):
            yy = y + 18 + i * 22
            rounded(draw, (x + 12 + i * 18, yy, x + w - 16 + i * 3, yy + 12), 6, theme["soft"], theme["line"])
        draw.line([x + 28, y + h - 28, x + w - 28, y + 24], fill=theme["accent"], width=3)


def draw_slide(draw, box, num, title, body, theme):
    x, y, w, h = box
    rounded(draw, (x, y, x + w, y + h), 14, theme["slide"], theme["line"], 2)
    draw.rectangle((x, y, x + 8, y + h), fill=theme["primary"])
    draw.text((x + 20, y + 16), f"{num:02d}", fill=theme["primary"], font=F_NUM)
    draw.text((x + 48, y + 13), title, fill=theme["ink"], font=F_SLIDE_TITLE)
    draw.line((x + 20, y + 48, x + w - 20, y + 48), fill=theme["line"], width=1)
    draw.multiline_text((x + 22, y + 60), body, fill=theme["muted"], font=F_SLIDE_BODY, spacing=5)
    draw_icon(draw, x + 22, y + 112, w - 44, h - 138, theme, num)
    draw.rectangle((x + 22, y + h - 22, x + w - 22, y + h - 18), fill=theme["soft"])
    draw.rectangle((x + 22, y + h - 22, x + 22 + (w - 44) * num / 20, y + h - 18), fill=theme["accent"])


def render(theme):
    width, height = 2400, 1600
    img = Image.new("RGB", (width, height), theme["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((80, 48), theme["title"], fill=theme["ink"], font=F_TITLE)
    draw.text(
        (80, 105),
        "20分钟 / 20页组会汇报 · 开放词汇SGG · Novel Predicate Semantic Transfer",
        fill=theme["muted"],
        font=F_SUB,
    )
    draw.line((80, 145, width - 80, 145), fill=theme["line"], width=2)

    cols, rows = 5, 4
    margin_x, margin_y = 80, 190
    gap_x, gap_y = 28, 30
    slide_w = (width - margin_x * 2 - gap_x * (cols - 1)) / cols
    slide_h = (height - margin_y - 60 - gap_y * (rows - 1)) / rows

    for i, (title, body) in enumerate(SLIDES, start=1):
        col = (i - 1) % cols
        row = (i - 1) // cols
        x = margin_x + col * (slide_w + gap_x)
        y = margin_y + row * (slide_h + gap_y)
        draw_slide(draw, (x, y, slide_w, slide_h), i, title, body, theme)

    out = OUT / f"{theme['name']}.png"
    img.save(out)
    return out


def main():
    for theme in THEMES:
        print(render(theme))


if __name__ == "__main__":
    main()
