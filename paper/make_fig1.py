"""Figure 1, the pipeline schematic -> paper/figures/fig1_pipeline.{svg,pdf,png}.

    python paper/make_fig1.py

Drawn in millimetres on a 183 mm x 84 mm canvas, in the same style and colors as paper/make_figures.py. One real moment
from the corpus runs through every stage: 018-078 (January 13, 1972, telephone), where Haldeman gives his name. Parakeet
wrote "alderman" (confidence 0.002), Whisper "haldeman"; the merged layer attaches Whisper's readings ("it" for "this",
"haldeman" for "alderman"); Argon's edit is alderman -> haldeman, reason "whisper"; the classifier names the segments
Haldeman. Every word, time and label below is copied from the corpus files (paper/check_quotes.py checks the text);
words are shown in proper case (the layers store them lowercased) for legibility.
Connectors run horizontally and vertically only, and every arrowhead points right. The measured results in the bottom
strip are read from the evaluation outputs.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

import make_figures as mf  # noqa: E402  (shared rcParams, colors, save, data)

W, H = 183.0, 84.0
INK, GRAY, MID, LIGHT = mf.INK, mf.GRAY, "#5f5f5f", "#e6e6e6"
PK, PK_FILL = "#2a78d6", "#e7f0fb"        # Parakeet: blue stroke on a pale tint
WH, WH_FILL = "#eb6834", "#fdeae1"        # Whisper
CO, CO_FILL = "#1baf7a", "#e2f5ee"        # correction / released corpus
VP, VP_FILL = "#e87ba4", "#fbe8f0"        # voiceprints
FONT = FontProperties(family="Arial")  # findfont falls back to a default sans if Arial is missing
SMALL = 5.6                               # smallest size drawn: 5 pt once 183 mm is scaled to the 165 mm text width


def tw(s, size):
    """Width of a string in mm at a font size in points, measured from the font itself."""
    if not s.strip():  # a blank string has no outline to measure
        return tw("a" + s + "a", size) - tw("aa", size)
    return TextPath((0, 0), s, size=size, prop=FONT).get_extents().width * 25.4 / 72


fig = plt.figure(figsize=(W * mf.MM, H * mf.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def text(x, y, s, size=6.0, color=INK, weight="normal", ha="left", va="center", style="normal", z=6):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style, zorder=z)


def box(x, y, w, h, fc="white", ec=LIGHT, lw=0.6, r=1.2, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def chip(x, y, s, ec, fc, dashed=False, size=SMALL, h=3.4):
    """A word chip with its left edge at x and its centre at y; returns its right edge."""
    w = tw(s, size) + 2.0
    box(x, y - h / 2, w, h, fc=fc, ec=ec, lw=0.55, r=0.9, ls=(0, (1.6, 1.0)) if dashed else "-", z=3)
    text(x + w / 2, y, s, size=size, ha="center", color=INK)
    return x + w


def chips(x, y, words, ec, fc, gap=0.8, dashed=()):
    for i, wd in enumerate(words):
        x = chip(x, y, wd, ec, fc, dashed=i in dashed) + gap
    return x - gap


def route(points, color=GRAY, lw=0.7, head=True):
    """A connector through `points` with horizontal and vertical segments only; the arrowhead, if any, points right."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        assert x0 == x1 or y0 == y1, f"diagonal segment {points}"
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="butt", solid_joinstyle="miter")
    if head:
        (xa, ya), (xb, yb) = points[-2], points[-1]
        assert ya == yb and xb > xa, f"arrowhead must point right: {points}"
        L, Hh = 1.5, 0.55
        ax.add_patch(Polygon([(xb, yb), (xb - L, yb + Hh), (xb - L, yb - Hh)], closed=True, fc=color, ec=color, lw=0.3,
                             zorder=3))


def step(x, y, n, label):
    ax.add_patch(plt.Circle((x + 1.9, y), 1.9, color=INK, zorder=4))
    text(x + 1.9, y - 0.05, str(n), size=6.0, color="white", weight="bold", ha="center")
    text(x + 4.8, y, label, size=6.9, weight="bold")


def doc(x, y, w=6.0, h=7.6, color=GRAY):
    f = 1.8
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)], closed=True,
                         fc="white", ec=color, lw=0.6, zorder=3))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)], closed=True, fc=LIGHT,
                         ec=color, lw=0.5, zorder=4))
    for k in range(4):
        yy = y + h - 2.9 - k * 1.25
        ax.plot([x + 1.0, x + w - (1.0 if k else 2.2)], [yy, yy], color=LIGHT if k else "#bdbdbd", lw=0.7, zorder=4,
                solid_capstyle="round")


# ---- stage headers -----------------------------------------------------------------------------------------
TOP = 80.5
X2, X3, X4, X5 = 36.0, 84.0, 124.0, 157.5
step(2.0, TOP, 1, "Audio")
step(X2, TOP, 2, "Two recognizers and voiceprints")
step(X3, TOP, 3, "Merge")
step(X4, TOP, 4, "Archival correction")
step(X5, TOP, 5, "Corpus")

# ---- 1 audio -----------------------------------------------------------------------------------------------
cx, cy = 15.5, 50.0
n = 46
for i in range(n):
    t = i / (n - 1)
    env = 0.25 + 0.75 * math.sin(math.pi * t) ** 0.6
    amp = env * (0.45 + 0.55 * abs(math.sin(i * 1.7) * math.cos(i * 0.53)))
    x = 3.0 + t * 25.0
    ax.plot([x, x], [cy - 7.5 * amp, cy + 7.5 * amp], color="#7d7d7d", lw=0.55, solid_capstyle="round")
text(cx, 38.6, "White House tapes", size=6.5, weight="bold", ha="center")
text(cx, 35.6, "19,426 conversations", size=5.8, color=MID, ha="center")
text(cx, 33.0, "2,206 hours, 1971–73", size=5.8, color=MID, ha="center")
text(cx, 30.4, "withdrawals muted", size=5.8, color=MID, ha="center")

# ---- 2 recognizers -----------------------------------------------------------------------------------------
rows = {"pk": 64.5, "wh": 46.0, "vp": 25.0}
text(X2, rows["pk"] + 7.4, "Parakeet TDT 0.6B, fine-tuned", size=6.5, weight="bold")
text(X2, rows["pk"] + 4.6, "word timings and confidence", size=5.8, color=MID)
end_pk = chips(X2, rows["pk"], ["This", "is", "Bob", "Alderman"], PK, PK_FILL, dashed=(3,))
text(X2 + 14.0, rows["pk"] - 3.7, "dashed: low confidence", size=SMALL, color=PK, style="italic")

text(X2, rows["wh"] + 7.4, "Whisper large-v3, unchanged", size=6.5, weight="bold")
text(X2, rows["wh"] + 4.6, "an independent second hearing", size=5.8, color=MID)
end_wh = chips(X2, rows["wh"], ["It's", "Bob", "Haldeman"], WH, WH_FILL)

text(X2, rows["vp"] + 7.4, "Voiceprint classifier", size=6.5, weight="bold")
text(X2, rows["vp"] + 4.6, "14 voices, NARA's list of who was present", size=5.8, color=MID)
segs = [("Haldeman", 15.0, VP), ("Haldeman", 15.0, VP), ("Haldeman", 11.0, VP), ("?", 6.0, LIGHT)]
x = X2
for name, w, col in segs:
    ax.add_patch(Rectangle((x, rows["vp"] - 1.7), w - 0.5, 3.4, fc=col, ec="none", zorder=3))
    text(x + (w - 0.5) / 2, rows["vp"], name, size=SMALL, color="white" if col == VP else INK, ha="center")
    x += w
end_vp = x - 0.5

# audio -> the three rows: out of the waveform, down a trunk, right into each row
TRUNK = 31.5
for y in rows.values():
    route([(29.0, cy), (TRUNK, cy), (TRUNK, y), (X2 - 1.0, y)])

# ---- 3 merge -----------------------------------------------------------------------------------------------
YM = 55.0
text(X3, YM + 12.2, "Parakeet's words kept;", size=5.8, color=MID)
text(X3, YM + 9.6, "Whisper's reading attached", size=5.8, color=MID)


def with_alt(x, top, alt, dashed=False):
    """Parakeet's word with Whisper's reading attached below it; returns the right edge."""
    w = max(tw(top, SMALL), tw(alt, SMALL)) + 2.0
    box(x, YM - 1.7, w, 3.4, fc=PK_FILL, ec=PK, lw=0.55, r=0.9, ls=(0, (1.6, 1.0)) if dashed else "-", z=3)
    text(x + w / 2, YM, top, size=SMALL, ha="center")
    box(x, YM - 5.9, w, 3.4, fc=WH_FILL, ec=WH, lw=0.55, r=0.9, z=3)
    text(x + w / 2, YM - 4.2, alt, size=SMALL, ha="center")
    ax.plot([x + w / 2] * 2, [YM - 1.7, YM - 2.5], color=GRAY, lw=0.5, zorder=2)
    return x + w


x = with_alt(X3, "This", "It's") + 0.8
x = chips(x, YM, ["is", "Bob"], PK, PK_FILL) + 0.8
end_m = with_alt(x, "Alderman", "Haldeman", dashed=True)
# Parakeet's row feeds the merged words, Whisper's row the attached readings
XJ = X3 - 3.5
route([(end_pk + 1.0, rows["pk"]), (XJ, rows["pk"]), (XJ, YM), (X3 - 1.0, YM)])
route([(end_wh + 1.0, rows["wh"]), (XJ, rows["wh"]), (XJ, YM - 4.2), (X3 - 1.0, YM - 4.2)])

# ---- 4 correction ------------------------------------------------------------------------------------------
Y4 = 40.0
BW = 28.6
labels = ["NARA\nsubject log", "Haldeman's\ndiary", "Same-day\nrecordings"]
widths = [max(tw(t, SMALL) for t in lab.split("\n")) for lab in labels]
c0, c2 = X4 - 2.0 + widths[0] / 2, X4 + BW + 2.0 - widths[2] / 2       # outer labels overhang the box by 2 mm
centres = [c0, (c0 + c2) / 2, c2]
assert X4 + 1.0 < c0 and c2 < X4 + BW - 1.0, "a feed line would miss the model's box"
for k in range(2):
    assert centres[k] + widths[k] / 2 + 0.8 < centres[k + 1] - widths[k + 1] / 2, "document labels collide"
for dcx, lab in zip(centres, labels):
    doc(dcx - 3.0, 60.5)
    text(dcx, 56.6, lab, size=SMALL, color=MID, ha="center", va="center")
    ax.plot([dcx, dcx], [53.8, Y4 + 9.0], color=GRAY, lw=0.6, zorder=2)   # feeds the box from above (no head)
box(X4, Y4, BW, 9.0, fc=CO_FILL, ec=CO, lw=0.7, r=1.4, z=3)
text(X4 + BW / 2, Y4 + 5.9, "Language model", size=6.5, weight="bold", ha="center")
text(X4 + BW / 2, Y4 + 2.9, "returns an edit list", size=5.8, color=MID, ha="center")
XK = X4 - 3.6
route([(end_m + 0.8, YM), (XK, YM), (XK, Y4 + 4.5), (X4 - 0.6, Y4 + 4.5)])
# the edit it returns
ey = 32.6
box(X4, ey - 3.6, BW, 7.2, fc="white", ec=CO, lw=0.55, r=0.9, z=3)
x1 = X4 + 1.4
text(x1, ey + 1.4, "Alderman", size=SMALL, color=INK)
x2 = x1 + tw("Alderman", SMALL) + 1.0
text(x2, ey + 1.4, "→", size=6.0, color=CO)
text(x2 + tw("→", 6.0) + 1.0, ey + 1.4, "Haldeman", size=SMALL, color=INK, weight="bold")
text(X4 + 1.4, ey - 1.8, "reason: whisper", size=SMALL, color=MID)
ax.plot([X4 + BW / 2, X4 + BW / 2], [Y4, ey + 3.6], color=CO, lw=0.6, zorder=2)

# ---- 5 corpus ----------------------------------------------------------------------------------------------
Yc, Wc, Hc = 31.0, 183.0 - 1.5 - X5, 34.0
box(X5, Yc, Wc, Hc, fc="white", ec="#bdbdbd", lw=0.6, r=1.4, z=2)
# the opening of 018-078 as released (the .txt line "[00:00:00] HALDEMAN: Hello. Mr. President? This is Bob haldeman
# Oh, general hughes ..."); edits underlined: haldeman (reason whisper), general hughes (reason nara)
ly = Yc + Hc - 4.0
text(X5 + 1.2, ly, "00:00", size=SMALL, color=GRAY)
ax.add_patch(plt.Circle((X5 + 7.7, ly), 0.8, color=VP, zorder=4))
text(X5 + 9.2, ly, "Haldeman", size=5.8, color=VP, weight="bold")
lines = [[("Hello.", None), ("Mr.", None), ("President?", None)],
         [("This", None), ("is", None), ("Bob", None), ("Haldeman", CO)],
         [("Oh,", None), ("General", CO), ("Hughes", CO)]]
for words in lines:
    ly -= 4.4
    wx = X5 + 1.6
    for k, (wd, mark) in enumerate(words):
        text(wx, ly, wd, size=SMALL)
        wlen = tw(wd, SMALL)
        nxt = wx + wlen + tw(" ", SMALL) + 0.25
        if mark:  # one underline per edit, joined across its words
            end = nxt if k + 1 < len(words) and words[k + 1][1] else wx + wlen
            ax.plot([wx, end], [ly - 1.25, ly - 1.25], color=mark, lw=0.8, zorder=4)
        wx = nxt
    assert wx - 0.25 - tw(" ", SMALL) < X5 + Wc - 0.8, "transcript line overflows the box"
text(X5 + Wc / 2, Yc + 7.6, "19.0 M words", size=5.8, color=MID, ha="center")
text(X5 + Wc / 2, Yc + 4.8, "word timings · speakers", size=SMALL, color=MID, ha="center")
text(X5 + Wc / 2, Yc + 2.2, "raw layers · edit list", size=SMALL, color=MID, ha="center")
text(X5 + Wc / 2, Yc + Hc + 3.6, "Released transcripts", size=6.5, weight="bold", ha="center")
route([(X4 + BW + 0.6, Y4 + 4.5), (X5 - 0.6, Y4 + 4.5)])
# speakers run along the bottom, up beside the corpus box and into it from the left
XV = X5 - 2.6
YV = Yc + 6.0
route([(end_vp + 1.0, rows["vp"]), (XV, rows["vp"]), (XV, YV), (X5 - 0.6, YV)], color=VP)
text((end_vp + X4) / 2, rows["vp"] - 2.0, "a speaker label for each segment", size=5.8, color=VP, ha="center")

# ---- bottom strip: what each stage contributes, measured (read from the evaluation outputs) -----------------------
A = mf.D["accuracy"]["test"]
ft = mf.D["finetune"]
sp = json.loads((mf.DATA_DIR / "frus_test" / "results.json").read_text(encoding="utf-8"))["speakers"]
items = [(f"{ft['Parakeet v3, stock'][0]:.1f}% → {ft['v5'][0]:.1f}%", "fine-tuning, train rooms", PK, PK_FILL),
         (f"{A['whisper']['recall'] * 100:.1f}% → {A['corrected']['recall'] * 100:.1f}%",
          "held-out test, word recall", CO, CO_FILL),
         (f"{A['whisper']['names'] * 100:.1f}% → {A['corrected']['names'] * 100:.1f}%",
          "held-out test, name recall", CO, CO_FILL),
         (f"{sp['named_precision'] * 100:.1f}%", "speaker names correct, held-out test", VP, VP_FILL)]
ax.plot([2, W - 2], [14.0, 14.0], color=LIGHT, lw=0.6)
text(2.0, 7.6, "Measured", size=6.0, weight="bold", color=INK)
x = X2
for head, sub, ec, fc in items:
    bw = max(tw(head, 6.9), tw(sub, 5.8)) + 3.2
    box(x, 2.6, bw, 9.6, fc=fc, ec=ec, lw=0.7, r=1.2, z=3)
    text(x + 1.6, 9.4, head, size=6.9, weight="bold")
    text(x + 1.6, 5.2, sub, size=5.8, color=INK)
    x += bw + 3.0
assert x - 3.0 < W - 1.5, "measured strip overflows"

mf.save(fig, "fig1_pipeline")
print("written", mf.OUT / "fig1_pipeline.pdf")
