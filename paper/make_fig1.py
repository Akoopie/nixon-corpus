"""Figure 1, the pipeline schematic -> paper/figures/fig1_pipeline.{svg,pdf,png}.

    python paper/make_fig1.py

Drawn in millimetres on a 183 mm x 86 mm canvas, in the same style and colors as paper/make_figures.py. One real moment
from the corpus runs through every stage: 018-078 (January 13, 1972, telephone), where Haldeman gives his name. Parakeet
wrote "alderman" (confidence 0.002), Whisper "it's bob haldeman"; the Whisper row shows only Whisper's readings that
differ from Parakeet's, each under the word it differs from, as the merged layer attaches them ("it's" for "this",
"haldeman" for "alderman"); Argon's edit is alderman -> haldeman, reason "whisper"; the classifier picks Haldeman among
its fourteen voices (the other party, General Hughes, has no class). Every word, time and label below is copied from the
corpus files (paper/check_quotes.py checks the text); words are shown in proper case (the layers store them
lowercased) for legibility. The audio column shows a waveform above a cartoon of the reel-to-reel tape the system
recorded on. Connectors run horizontally and vertically only, and every arrowhead points right. The bar strip at the
bottom is read from the evaluation outputs.
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

W, H = 183.0, 86.0
INK, GRAY, MID, LIGHT = mf.INK, mf.GRAY, "#5f5f5f", "#e6e6e6"
PK, PK_FILL = "#2a78d6", "#e7f0fb"        # Parakeet: blue stroke on a pale tint
WH, WH_FILL = "#eb6834", "#fdeae1"        # Whisper
CO, CO_FILL = "#1baf7a", "#e2f5ee"        # correction / released corpus
VP, VP_FILL = "#e87ba4", "#fbe8f0"        # voiceprints
LM, LM_SUB = "#6741c4", "#e4dbfa"         # the large language model: the one solid box
COOL = ["#2a78d6", "#16a3a0", "#58b4e6"]  # accent lines in the waveform
TAPE, REEL, REEL_FILL = "#7a5230", "#6f6f6f", "#f3f3f3"
FONT = FontProperties(family="Arial")  # findfont falls back to a default sans if Arial is missing
FONT_B = FontProperties(family="Arial", weight="bold")
SMALL = 5.6                               # smallest size drawn: 5 pt once 183 mm is scaled to the 165 mm text width


def tw(s, size, bold=False):
    """Width of a string in mm at a font size in points, measured from the font itself."""
    if not s.strip():  # a blank string has no outline to measure
        return tw("a" + s + "a", size, bold) - tw("aa", size, bold)
    return TextPath((0, 0), s, size=size, prop=FONT_B if bold else FONT).get_extents().width * 25.4 / 72


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


def chip(x, y, s, ec, fc, dashed=False, size=SMALL, h=3.4, w=None, color=INK, weight="normal"):
    """A word chip with its left edge at x and its centre at y; returns its right edge."""
    w = w or tw(s, size) + 2.0
    box(x, y - h / 2, w, h, fc=fc, ec=ec, lw=0.55, r=0.9, ls=(0, (1.6, 1.0)) if dashed else "-", z=3)
    text(x + w / 2, y, s, size=size, ha="center", color=color, weight=weight)
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


def reel(x, y, pack, R=5.0):
    """A cartoon tape reel: flange, wound tape, three windows, hub."""
    ax.add_patch(plt.Circle((x, y), R, fc=REEL_FILL, ec=REEL, lw=0.6, zorder=3))
    ax.add_patch(plt.Circle((x, y), pack, fc=TAPE, ec="none", zorder=4))
    for a in (90, 210, 330):
        t = math.radians(a)
        ax.add_patch(plt.Circle((x + 3.0 * math.cos(t), y + 3.0 * math.sin(t)), 0.95, fc=REEL_FILL, ec=REEL, lw=0.4,
                                zorder=5))
    ax.add_patch(plt.Circle((x, y), 1.4, fc=REEL_FILL, ec=REEL, lw=0.5, zorder=6))
    ax.add_patch(plt.Circle((x, y), 0.45, fc=REEL, ec="none", zorder=7))


# ---- stage headers -----------------------------------------------------------------------------------------
TOP = H - 3.5
X2, X3, X4, X5 = 36.0, 84.0, 122.5, 157.5
step(2.0, TOP, 1, "Audio")
step(X2, TOP, 2, "Two recognizers and voiceprints")
step(X3, TOP, 3, "Merge")
step(X4, TOP, 4, "Archival correction")
step(X5, TOP, 5, "Corpus")

# ---- 1 audio: the waveform, and below it the reel-to-reel tape it was recorded on ----------------------------
cx, cy = 15.5, 62.0
n = 46
for i in range(n):
    t = i / (n - 1)
    env = 0.25 + 0.75 * math.sin(math.pi * t) ** 0.6
    amp = env * (0.45 + 0.55 * abs(math.sin(i * 1.7) * math.cos(i * 0.53)))
    x = 3.0 + t * 25.0
    col = [GRAY, COOL[0], "#7d7d7d", COOL[1], COOL[2], "#7d7d7d"][i % 6]
    ax.plot([x, x], [cy - 6.5 * amp, cy + 6.5 * amp], color=col, lw=0.6, solid_capstyle="round")
RY, XL, XR = 46.0, 9.5, 21.5
reel(XL, RY, 4.4)   # supply reel, nearly full
reel(XR, RY, 2.7)   # take-up reel
TY = RY - 6.6       # tape path runs under the reels, over two guides and the head
ax.plot([XL, XL + 3.2, XR - 3.2, XR], [RY - 4.4, TY, TY, RY - 2.7], color=TAPE, lw=0.8, zorder=4,
        solid_joinstyle="round")
for gx in (XL + 3.2, XR - 3.2):
    ax.add_patch(plt.Circle((gx, TY - 0.75), 0.75, fc=REEL_FILL, ec=REEL, lw=0.5, zorder=3))
ax.add_patch(Rectangle((cx - 1.3, TY - 1.9), 2.6, 1.9, fc="#bdbdbd", ec=REEL, lw=0.5, zorder=3))
text(cx, 33.6, "White House tapes", size=6.5, weight="bold", ha="center")
text(cx, 30.6, "19,426 conversations", size=5.8, color=MID, ha="center")
text(cx, 28.0, "2,206 hours, 1971–73", size=5.8, color=MID, ha="center")
text(cx, 25.4, "withdrawals muted", size=5.8, color=MID, ha="center")

# ---- 2 recognizers -----------------------------------------------------------------------------------------
rows = {"pk": 66.5, "wh": 48.0, "vp": 28.5}
text(X2, rows["pk"] + 7.4, "Parakeet TDT 0.6B, fine-tuned", size=6.5, weight="bold")
text(X2, rows["pk"] + 4.6, "word timings and confidence", size=5.8, color=MID)
end_pk = chips(X2, rows["pk"], ["This", "is", "Bob", "Alderman"], PK, PK_FILL, dashed=(3,))
text(X2 + 14.0, rows["pk"] - 3.7, "dashed: low confidence", size=SMALL, color=PK, style="italic")

text(X2, rows["wh"] + 7.4, "Whisper large-v3, unchanged", size=6.5, weight="bold")
text(X2, rows["wh"] + 4.6, "independent hearing; differences shown", size=5.8, color=MID)
# each reading under the Parakeet word it differs from: "It's" under "This", "Haldeman" under "Alderman"
w_name = max(tw("Alderman", SMALL), tw("Haldeman", SMALL)) + 2.0
x_name = end_pk - (tw("Alderman", SMALL) + 2.0)
chip(X2, rows["wh"], "It's", WH, WH_FILL)
end_wh = chip(x_name, rows["wh"], "Haldeman", WH, WH_FILL, w=w_name)

text(X2, rows["vp"] + 7.4, "Voiceprint classifier", size=6.5, weight="bold")
text(X2, rows["vp"] + 4.6, "14 known voices, narrowed to NARA's list of who was present", size=5.8, color=MID)
x = X2
for name in ["Nixon", "Haldeman", "Kissinger", "Ehrlichman"]:
    picked = name == "Haldeman"
    x = chip(x, rows["vp"], name, VP, VP if picked else VP_FILL, color="white" if picked else INK,
             weight="bold" if picked else "normal") + 0.8
text(x + 0.4, rows["vp"], "+ 10 more", size=SMALL, color=MID)
end_vp = x + 0.4 + tw("+ 10 more", SMALL)

# audio -> the three rows: out of the waveform, down a trunk, right into each row
TRUNK = 31.5
for y in rows.values():
    route([(29.0, cy), (TRUNK, cy), (TRUNK, y), (X2 - 1.0, y)])

# ---- 3 merge -----------------------------------------------------------------------------------------------
YM = 57.0
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
assert end_wh + 2.0 < XJ, "Whisper row runs into the merge junction"
route([(end_pk + 1.0, rows["pk"]), (XJ, rows["pk"]), (XJ, YM), (X3 - 1.0, YM)])
route([(end_wh + 1.0, rows["wh"]), (XJ, rows["wh"]), (XJ, YM - 4.2), (X3 - 1.0, YM - 4.2)])

# ---- 4 correction: the large language model, the one solid box -------------------------------------------------
Y4, H4 = 40.5, 12.0
T1, T2 = "Large Language Model", "Gemini 4 Argon"
BW = max(tw(T1, 7.0, bold=True), tw(T2, 6.2)) + 4.0
labels = ["NARA\nsubject log", "Haldeman's\ndiary", "Same-day\nrecordings"]
widths = [max(tw(t, SMALL) for t in lab.split("\n")) for lab in labels]
c0, c2 = X4 - 2.0 + widths[0] / 2, X4 + BW + 2.0 - widths[2] / 2       # outer labels overhang the box by 2 mm
centres = [c0, (c0 + c2) / 2, c2]
assert X4 + 1.0 < c0 and c2 < X4 + BW - 1.0, "a feed line would miss the model's box"
for k in range(2):
    assert centres[k] + widths[k] / 2 + 0.8 < centres[k + 1] - widths[k + 1] / 2, "document labels collide"
for dcx, lab in zip(centres, labels):
    doc(dcx - 3.0, 63.0)
    text(dcx, 59.4, lab, size=SMALL, color=MID, ha="center", va="center")
    ax.plot([dcx, dcx], [56.8, Y4 + H4], color=GRAY, lw=0.6, zorder=2)   # feeds the box from above (no head)
box(X4, Y4, BW, H4, fc=LM, ec=LM, lw=0.9, r=1.6, z=3)
text(X4 + BW / 2, Y4 + H4 / 2 + 2.0, T1, size=7.0, weight="bold", color="white", ha="center")
text(X4 + BW / 2, Y4 + H4 / 2 - 2.4, T2, size=6.2, color=LM_SUB, ha="center")
YL = Y4 + H4 / 2
XK = X4 - 3.6
assert end_m + 1.8 < XK, "merge words run into the correction feed"
route([(end_m + 0.8, YM), (XK, YM), (XK, YL), (X4 - 0.6, YL)])
# the edit it returns, reached by one arrow from the model
EX, ey = X4 + 5.0, 35.0
box(EX, ey - 3.6, X4 + BW - EX, 7.2, fc="white", ec=CO, lw=0.55, r=0.9, z=3)
x1 = EX + 1.4
text(x1, ey + 1.4, "Alderman", size=SMALL, color=INK)
x2 = x1 + tw("Alderman", SMALL) + 1.0
text(x2, ey + 1.4, "→", size=6.0, color=CO)
x3 = x2 + tw("→", 6.0) + 1.0
text(x3, ey + 1.4, "Haldeman", size=SMALL, color=INK, weight="bold")
assert x3 + tw("Haldeman", SMALL, bold=True) < X4 + BW - 0.8, "edit text overflows its box"
text(x1, ey - 1.8, "reason: whisper", size=SMALL, color=MID)
route([(X4 + 2.2, Y4), (X4 + 2.2, ey), (EX - 0.6, ey)], color=LM, lw=0.8)

# ---- 5 corpus ----------------------------------------------------------------------------------------------
Yc, Wc, Hc = 33.0, 183.0 - 1.5 - X5, 34.0
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
assert X4 + BW + 2.6 < X5, "no room for the arrow into the corpus"
route([(X4 + BW + 0.6, YL), (X5 - 0.6, YL)])
# speakers run along the bottom, up beside the corpus box and into it from the left
XV = X5 - 2.6
YV = Yc + 6.0
assert X4 + BW + 1.2 < XV, "speaker line touches the edit box"
route([(end_vp + 1.0, rows["vp"]), (XV, rows["vp"]), (XV, YV), (X5 - 0.6, YV)], color=VP)
text((end_vp + X4) / 2 + 3.0, rows["vp"] - 2.0, "a speaker label for each segment", size=5.8, color=VP, ha="center")

# ---- bottom strip: the measured result, as bars (read from the evaluation outputs) ------------------------------
A = mf.D["accuracy"]["test"]
sp = json.loads((mf.DATA_DIR / "frus_test" / "results.json").read_text(encoding="utf-8"))["speakers"]
SY = 20.0
ax.plot([2, W - 2], [SY, SY], color=LIGHT, lw=0.6)
text(2.0, SY - 3.0, "Measured on 165 conversations transcribed by the State Department's historians, "
     "held out from all development", size=6.2, weight="bold")
RW, RC = 9.4, 4.6          # bar rows: Whisper alone, Nixon Corpus
text(X2 - 2.0, RW, "Whisper alone", size=5.8, color=INK, ha="right")
text(X2 - 2.0, RC, "Nixon Corpus", size=5.8, color=INK, ha="right", weight="bold")
panels = [("Words recovered", A["whisper"]["recall"], A["corrected"]["recall"], CO),
          ("Names recovered", A["whisper"]["names"], A["corrected"]["names"], CO),
          ("Speaker labels correct (130 conversations)", None, sp["named_precision"], VP)]
PW, PG = 45.0, 5.0
BM = PW - 9.0              # bar length for 100%
for k, (title, vw, vc, col) in enumerate(panels):
    x0 = X2 + k * (PW + PG)
    text(x0, 13.6, title, size=5.8, color=MID)
    for y, v, c, bold in ((RW, vw, WH, False), (RC, vc, col, True)):
        ax.add_patch(Rectangle((x0, y - 1.5), BM, 3.0, fc="#f1f1f1", ec="none", zorder=2))   # the 100% track
        if v is None:
            text(x0 + 1.0, y, "no speaker labels", size=SMALL, color=GRAY, style="italic")
            continue
        ax.add_patch(Rectangle((x0, y - 1.5), BM * v, 3.0, fc=c, ec="none", zorder=3))
        text(x0 + BM * v + 1.0, y, f"{v * 100:.1f}%", size=6.2, weight="bold" if bold else "normal", z=7)
assert X2 + 3 * PW + 2 * PG <= W - 1.5, "bar strip overflows"

mf.save(fig, "fig1_pipeline")
print("written", mf.OUT / "fig1_pipeline.pdf")
