"""The paper's figures 2-4, from paper/data/figure_data.json and the evaluation outputs -> paper/figures/*.{svg,pdf,png}.

    python paper/figure_data.py && python paper/speaker_time.py && python paper/make_figures.py

    fig2_accuracy    held-out test per conversation, accuracy by layer, fine-tuning by round, accuracy by location
    fig3_artifacts   two Whisper-only artifacts shown in the text (loop, text over silence) and three rates
    fig4_corpus      hours by month and location, speech per named voice, speaker accuracy on the test set

Style follows Nature Communications: Arial, 5-7 pt at final size, bold lowercase panel letters, thin axes with outward
ticks, no grid, individual data points where they exist, width 183 mm (two columns). Key results sit in boxed callouts;
every number in them is read from the data files, never typed. Figure 1 (the pipeline) is drawn by paper/make_fig1.py.
"""
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from paths import DATA_DIR  # noqa: E402

D = json.loads((ROOT / "data/figure_data.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

MM = 1 / 25.4
W2, W1 = 183 * MM, 89 * MM
# Layers of the corpus, one color each in every figure (validated categorical slots; Parakeet and the merge share a
# hue at two lightness steps because the merge is Parakeet's text with Whisper's readings attached)
C = {"whisper": "#eb6834", "parakeet_v5": "#86b6ef", "merged": "#2a78d6", "corrected": "#1baf7a"}
TINT = {"whisper": "#fdeae1", "parakeet_v5": "#eef5fd", "merged": "#e7f0fb", "corrected": "#e2f5ee", "gray": "#f2f2f2"}
LAYER_LABEL = {"whisper": "Whisper large-v3 alone", "parakeet_v5": "Parakeet v5 alone",
               "merged": "Merged (no language model)", "corrected": "Corrected layer"}
GRAY, LIGHT, INK = "#8c8c8c", "#d9d9d9", "#222222"
LOCS = [("Oval Office", "#2a78d6"), ("White House telephone", "#eb6834"), ("Executive Office Building", "#1baf7a"),
        ("Camp David", "#eda100"), ("Cabinet Room", "#e87ba4")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.2, "axes.edgecolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "axes.titleweight": "normal",
    "axes.titlelocation": "left", "axes.titlepad": 4,
})


def save(fig, name):
    place_letters(fig)
    # no creation dates in the files: they carry the local time zone
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, metadata={"Date": None})
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


LETTERS = []  # (fig, ax, letter), placed by save() once the layout is final


def letter(ax, s):
    """Queue a bold panel letter at the top left of the axes, outside its tick labels (Nature style)."""
    LETTERS.append((ax.figure, ax, s))


def place_letters(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for _, ax, s in [x for x in LETTERS if x[0] is fig]:
        bb = ax.get_tightbbox(r)
        x0 = inv.transform((bb.x0, 0))[0]
        y1 = inv.transform((0, ax.get_window_extent(r).y1))[1]
        if ax.get_title():
            y1 = max(y1, inv.transform((0, ax.title.get_window_extent(r).y1))[1] - 0.012)
        fig.text(x0, y1 + 0.012, s, fontsize=8, fontweight="bold", va="bottom", ha="left")


def callout(ax, xy, head, sub=None, key="corrected", align=(0, 1), coords="axes fraction", size=7.2):
    """A boxed key result: the numbers in bold, an optional plain line under them."""
    kids = [TextArea(head, textprops=dict(fontsize=size, fontweight="bold", color=INK))]
    if sub:
        kids.append(TextArea(sub, textprops=dict(fontsize=6.0, color=INK)))
    ab = AnnotationBbox(VPacker(children=kids, align="left", pad=0, sep=1.2), xy, xycoords=coords, box_alignment=align,
                        frameon=True, pad=0.35, annotation_clip=False, zorder=10,
                        bboxprops=dict(boxstyle="round,pad=0.35,rounding_size=0.6", fc=TINT[key], ec=C.get(key, GRAY),
                                       lw=0.7))
    ax.add_artist(ab)
    return ab


def released_text(conv: str, year: str, t0: float, t1: float) -> str:
    """The readable (cased, punctuated) text of release v1.1 for the word rows starting in [t0, t1] s."""
    import gzip
    import tarfile
    tar = DATA_DIR / "release" / "nixon-corpus-v1.1" / f"nixon-corpus-v1.1-conversations-{year}.tar"
    with tarfile.open(tar) as t:
        rec = json.load(gzip.open(t.extractfile(f"{conv}.json.gz")))
    return " ".join(w[0] for w in rec["words"] if t0 <= w[1] <= t1)


def pct(v):
    return f"{v:.1f}%"


# ---------------------------------------------------------------------------------------------------------------
def fig_accuracy():
    """Held-out test per conversation (a), accuracy of each layer (b), fine-tuning by round (c), by location (d)."""
    A = D["accuracy"]
    fig = plt.figure(figsize=(W2, 84 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 0.8], wspace=0.40, hspace=0.62,
                          left=0.07, right=0.98, bottom=0.10, top=0.92)

    # a: every test conversation
    a = fig.add_subplot(gs[0, 0])
    rows = D["test_per_conv"]
    xs = [float(r["whisper_recall"]) * 100 for r in rows]
    ys = [float(r["corrected_recall"]) * 100 for r in rows]
    ss = [4 + 26 * math.sqrt(int(r["ref_words"]) / 15000) for r in rows]
    a.plot([0, 100], [0, 100], color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    a.scatter(xs, ys, s=ss, color=C["corrected"], alpha=0.55, edgecolor="white", linewidth=0.4, zorder=2)
    above = sum(y > x for x, y in zip(xs, ys))
    callout(a, (0.0, 1.04), f"{above} of {len(rows)} conversations", "improve on Whisper alone", align=(0, 0))
    a.text(88, 75, "y = x", fontsize=6.0, color=GRAY, ha="left", rotation=45, rotation_mode="anchor")  # below the line, where no point lies
    a.set_xlim(15, 100)
    a.set_ylim(15, 100)
    a.set_aspect("equal")
    a.set_xlabel("Whisper alone, word recall (%)")
    a.set_ylabel("Corrected, word recall (%)")
    for n in (500, 5000, 15000):
        a.scatter([], [], s=4 + 26 * math.sqrt(n / 15000), color=C["corrected"], alpha=0.55, label=f"{n:,}")
    a.legend(title="Reference words", title_fontsize=6.0, loc="lower right", fontsize=6.0, labelspacing=0.3,
             handletextpad=0.2, borderpad=0.2)
    letter(a, "a")

    # b: each layer, development and test, with the test result boxed above each half
    gb = gs[0, 1].subgridspec(1, 2, wspace=0.12)
    layers = ["whisper", "parakeet_v5", "merged", "corrected"]
    for j, (metric, title, lo, hi, what) in enumerate((("recall", "Word recall (%)", 50, 85, "word recall"),
                                                       ("names", "Name recall (%)", 40, 90, "name recall"))):
        ax = fig.add_subplot(gb[j])
        for i, L in enumerate(layers):
            y = len(layers) - 1 - i
            t = A["test"][L][metric] * 100
            ci = A["test_ci"][L].get(metric)
            d = A["dev"][L][metric] * 100
            if ci:
                ax.plot([ci[0] * 100, ci[1] * 100], [y + 0.12] * 2, color=C[L], lw=1.2, solid_capstyle="butt")
            ax.plot(t, y + 0.12, "o", ms=4.2, color=C[L], mec="white", mew=0.5)
            ax.plot(d, y - 0.16, "o", ms=3.8, mfc="white", mec=C[L], mew=0.9)
        ax.set_yticks(range(len(layers)), [LAYER_LABEL[L] for L in layers[::-1]] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(layers) - 0.4)
        ax.set_xlabel(title)
        cw, ww = A["test"]["corrected"][metric] * 100, A["test"]["whisper"][metric] * 100
        callout(ax, (0.0, 1.04), f"{pct(cw)} vs {pct(ww)}", f"test {what}: corrected vs Whisper", align=(0, 0))
        if j == 0:
            letter(ax, "b")
        else:
            ax.legend([Line2D([], [], marker="o", ls="", color=GRAY, mec="white", ms=4.2),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, ms=3.8)],
                      ["Test (95% CI)", "Train"], loc="upper right", handletextpad=0.2, fontsize=6.0,
                      borderpad=0.1)

    # c: fine-tuning by round, pooled over the six development room recordings (a narrow panel, so the rise reads)
    gcd = gs[1, :].subgridspec(1, 2, width_ratios=[0.62, 1.38], wspace=0.62)
    c = fig.add_subplot(gcd[0])
    rounds = ["Base", "v1", "v3", "v4", "v5"]
    pooled = {k: D["finetune"][m][0] for k, m in zip(rounds, ["Parakeet v3, stock", "v1", "v3", "v4", "v5"])}
    xs_ = list(range(len(rounds)))
    ys_ = [pooled[r] for r in rounds]
    c.plot(xs_, ys_, color=C["merged"], lw=1.6, marker="o", ms=4, mec="white", mew=0.6, zorder=3)
    for x_, v in zip(xs_, ys_):
        c.text(x_, v + 0.9, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2, color=C["merged"], fontweight="bold")
    c.set_xticks(xs_, rounds)
    c.set_xlim(-0.4, 4.4)
    c.set_ylim(60, 75)
    c.set_ylabel("Word recall (%)")
    c.set_xlabel("Fine-tuning round")
    callout(c, (0.0, 1.06), f"{pct(pooled['Base'])} → {pct(pooled['v5'])}",
            "fine-tuning, six train room recordings", key="merged", align=(0, 0))
    letter(c, "c")

    # d: by recording location
    d = fig.add_subplot(gcd[1])
    bl = D["test_by_location"]
    locs = sorted(bl, key=lambda k: bl[k]["corrected"]["recall"])
    for i, loc in enumerate(locs):
        w, r = bl[loc]["whisper"]["recall"] * 100, bl[loc]["corrected"]["recall"] * 100
        d.plot([w, r], [i, i], color=LIGHT, lw=2.2, solid_capstyle="round", zorder=1)
        d.plot(w, i, "o", ms=4.5, color=C["whisper"], mec="white", mew=0.5, zorder=2)
        d.plot(r, i, "o", ms=4.5, color=C["corrected"], mec="white", mew=0.5, zorder=3)
        d.text(r + 1.2, i, f"{r:.1f}", va="center", fontsize=6.2, color=C["corrected"], fontweight="bold")
    short = {"Executive Office Building": "EOB", "White House telephone": "Telephone"}
    d.set_yticks(range(len(locs)), [f"{short.get(loc, loc)} (n = {bl[loc]['n']})" for loc in locs])
    d.tick_params(axis="y", length=0)
    d.spines["left"].set_visible(False)
    d.set_xlim(40, 100)
    d.set_ylim(-0.6, len(locs) - 0.4)
    d.set_xlabel("Word recall on the test set (%)")
    d.set_title("Held-out test by recording location", fontsize=6.9)
    d.legend([Line2D([], [], marker="o", ls="", color=C["whisper"], mec="white", ms=4.5),
              Line2D([], [], marker="o", ls="", color=C["corrected"], mec="white", ms=4.5)],
             ["Whisper alone", "Corrected layer"], loc="lower right", handletextpad=0.2, fontsize=6.0, borderpad=0.1)
    letter(d, "d")
    save(fig, "fig2_accuracy")


# ---------------------------------------------------------------------------------------------------------------
def fig_artifacts():
    """Whisper-only artifacts: a loop (a) and text over silence (b) with the corpus text, and their rates over the
    whole corpus (c, e; dropped speech, d, for the WhisperX-decoded fifth), from ~/nixon-data/paper/whisper_artifacts.json
    (counts.by_mode, METHODS 7p)."""
    ex = {e["panel"]: e for e in D["artifacts"]["examples"]}
    modes = json.loads((DATA_DIR / "paper" / "whisper_artifacts.json").read_text(encoding="utf-8"))["counts"]["by_mode"]
    allm, bat = modes["all"], modes["batched"]   # bat: the trial fifth, decoded with WhisperX (dropped speech only)
    titles = {"a": "Repetition loop", "b": "Dropped speech", "c": "Text produced over silence"}
    place = {"Executive Office Building": "EOB"}
    # the corrected layer as released, in its readable cased form (the loop: 421-018, 3,935-3,953 s); b has no words
    ours = {"a": released_text(ex["a"]["conv"], "1973", 3935.0, 3952.3), "c": ex["c"]["ours"]}
    assert ours["a"].startswith("But, John,") and ours["a"].endswith("same day"), ours["a"]
    ref_label = {"a": "Reference (WSPF)", "b": "Reference (WSPF)", "c": "Audio"}
    ref_text = {"a": ex["a"]["reference"], "b": ex["b"]["reference"],
                "c": "Withdrawn by NARA: a tone for the first 55 min 42 s, muted to silence before transcription."}
    whisper_text = {"a": ex["a"]["whisper"], "b": ex["b"]["whisper"],
                    "c": "you … you … you … (one word every 30 s, 111 times, 0:00–55:00)"}

    H = 83.0                               # figure height, mm
    fig = plt.figure(figsize=(W2, H * MM))
    fy = lambda mm: 1 - mm / H             # noqa: E731  (mm from the top -> figure fraction)
    LH, WRAP, X_LAB, X_TXT = 2.45, 122, 0.034, 0.15
    y = 1.0                                # mm from the top
    # two of the three examples: the loop and the text over silence
    for letter_, k in zip("ab", "ac"):
        e = ex[k]
        fig.text(0.0, fy(y), letter_, fontsize=8, fontweight="bold", va="top")
        fig.text(0.025, fy(y + 0.1), titles[k], fontsize=7.4, fontweight="bold", va="top")
        fig.text(0.99, fy(y + 0.3), f"{e['conv']} · {e['date']} · {place.get(e['place'], e['place'])}",
                 fontsize=6.2, color=GRAY, va="top", ha="right")
        y += 4.4
        for label, text, col in ((ref_label[k], ref_text[k], GRAY), ("Whisper alone", whisper_text[k], C["whisper"]),
                                 ("Corrected layer", ours[k], C["corrected"])):
            lines = textwrap.wrap(text.replace("—", "–"), WRAP) or [""]
            fig.add_artist(plt.Rectangle((0.025, fy(y + LH * len(lines) - 0.5)), 0.0035, (LH * len(lines) - 0.6) / H,
                                         transform=fig.transFigure, color=col, lw=0))
            fig.text(X_LAB, fy(y), label, fontsize=6.5, fontweight="bold", va="top")
            for j, line in enumerate(lines):
                fig.text(X_TXT, fy(y + j * LH), line, fontsize=6.6, va="top",
                         style="italic" if label in ("Whisper alone", "Corrected layer") else "normal")
            y += LH * len(lines) + 1.0
        y += 1.6

    # c, d: the two rates over the whole corpus, each headed by its count in a box
    y += 9.0
    gs = fig.add_gridspec(1, 3, left=0.105, right=0.975, bottom=9 / H, top=fy(y), wspace=0.95)
    panels = [
        ("c", (f"{allm['whisper_loops']:,} loops in Whisper's output", f"{allm['parakeet_v5_loops']:,} in Parakeet v5's"),
         "Loop words per 1,000 words",
         [("Whisper", 1000 * allm["whisper_loop_words"] / allm["whisper_words"], C["whisper"], 1.0),
          ("Parakeet v5", 1000 * allm["parakeet_v5_loop_words"] / allm["parakeet_words"], C["parakeet_v5"], 1.0)]),
        ("d", (f"{100 * bat['whisper_silent_parakeet_words'] / bat['parakeet_words']:.1f}% of Parakeet's words",
               "fall where WhisperX was silent ≥20 s"),
         "Other recognizer's words in this one's\n≥20 s silences (% of its words)",
         [("WhisperX (batched)", 100 * bat["whisper_silent_parakeet_words"] / bat["parakeet_words"], C["whisper"], 1.0),
          ("Parakeet v5", 100 * allm["parakeet_silent_whisper_words"] / allm["whisper_words"], C["parakeet_v5"], 1.0)]),
        ("e", (f"{allm['whisper_words_in_tones']:,} Whisper words over silence",
               f"in {allm['whisper_convs_text_in_tones']:,} conversations; Parakeet v5: {allm['parakeet_words_in_tones']:,}"),
         f"Words written over muted withdrawals\n(per hour; {allm['tone_hours']:.1f} hours of muted audio)",
         [("Whisper", allm["whisper_words_in_tones"] / allm["tone_hours"], C["whisper"], 1.0),
          ("Parakeet v5", allm["parakeet_words_in_tones"] / allm["tone_hours"], C["parakeet_v5"], 1.0)]),
    ]
    for i, (lt, (head, sub), xlabel, bars) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        vmax = max(v for _, v, _, _ in bars)
        for j, (lab, v, col, alpha) in enumerate(bars[::-1]):
            ax.barh(j, v, height=0.55, color=col, alpha=alpha, edgecolor="white", lw=0.5)
            ax.text(v + vmax * 0.03, j, f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 0.1 else f"{v:.3f}", va="center",
                    fontsize=6.5)
        ax.set_yticks(range(len(bars)), [b_[0] for b_ in bars[::-1]])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, vmax * 1.3)
        ax.set_xlabel(xlabel, fontsize=6.5)
        callout(ax, (1.0, 1.06), head, sub, key="whisper", align=(1, 0), size=6.9)
        letter(ax, lt)
    save(fig, "fig3_artifacts")


# ---------------------------------------------------------------------------------------------------------------
def fig_corpus():
    """Hours by month and location (a), speech per named voice (b), speaker accuracy on the held-out test (c)."""
    m = D["by_month_location"]
    st = json.loads((ROOT / "data/speaker_time.json").read_text(encoding="utf-8"))
    man = json.loads((DATA_DIR / "release" / "nixon-corpus-v1.1" / "MANIFEST.json").read_text(encoding="utf-8"))
    sp = json.loads((DATA_DIR / "frus_test" / "results.json").read_text(encoding="utf-8"))["speakers"]
    months = [k for k in m if "1971-02" <= k <= "1973-07"]
    fig = plt.figure(figsize=(W2, 74 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 0.85], wspace=0.55, left=0.06, right=0.955, bottom=0.25,
                          top=0.84)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))

    bottom = [0.0] * len(months)
    for name, color in LOCS:
        ys = [m[k].get(name, 0.0) for k in months]
        ax.bar(range(len(months)), ys, bottom=bottom, width=0.78, color=color, label=name, edgecolor="white", linewidth=0.3)
        bottom = [b_ + y for b_, y in zip(bottom, ys)]
    ax.set_ylabel("Audio hours per month")
    ticks = [i for i, k in enumerate(months) if k[5:] in ("01", "07") or i == 0]
    ax.set_xticks(ticks, [f"{'Jan' if months[i][5:] == '01' else 'Jul' if months[i][5:] == '07' else 'Feb'} {months[i][:4]}"
                          for i in ticks])
    ax.set_xlim(-0.7, len(months) - 0.3)
    ax.set_ylim(0, 150)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.13), columnspacing=1.0, handlelength=0.9,
              fontsize=6.2, borderpad=0.1)
    callout(ax, (0.0, 1.03), f"{man['conversations']:,} conversations · {man['audio_hours']:,.0f} hours",
            f"{man['words']:,} words (release {man['version']})", key="merged", align=(0, 0))
    letter(ax, "a")

    named = st["named"]
    names = sorted(named, key=lambda k: named[k]["hours"])
    unnamed = st["total_segment_hours"] - sum(v["hours"] for v in named.values())
    rows = [("No name", unnamed, LIGHT)] + [(k, named[k]["hours"], C["merged"]) for k in names]
    for i, (lab, h, col) in enumerate(rows):
        bx.barh(i, h, height=0.62, color=col, edgecolor="white", lw=0.4)
        bx.text(h * 1.12, i, f"{h:.0f}" if h >= 10 else f"{h:.1f}", va="center", fontsize=6)
    bx.set_xscale("log")
    bx.set_xlim(1, 3000)
    bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bx.set_yticks(range(len(rows)), [r[0] for r in rows])
    bx.tick_params(axis="y", length=0, labelsize=6.2)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Hours of speech (log scale)")
    bx.set_title("Hours of speech per named voice", fontsize=6.9)
    letter(bx, "b")

    ps = sp["per_speaker"]
    voices = sorted((k for k in names if k in ps and ps[k]["named"] >= 200), key=lambda k: ps[k]["named_right"] / ps[k]["named"])
    for i, k in enumerate(voices):
        v = 100 * ps[k]["named_right"] / ps[k]["named"]
        cx.plot([85, v], [i, i], color="#eeeeee", lw=2.2, solid_capstyle="round", zorder=1)
        cx.plot(v, i, "o", ms=4.5, color=C["corrected"], mec="white", mew=0.5, zorder=2)
        cx.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=6.0)
    cx.set_yticks(range(len(voices)), voices)
    cx.tick_params(axis="y", length=0, labelsize=6.2)
    cx.spines["left"].set_visible(False)
    cx.set_xlim(85, 101)
    cx.set_ylim(-0.6, len(voices) - 0.4)
    cx.set_xlabel("Named words correct (%)")
    callout(cx, (0.0, 1.03), f"{sp['named_precision'] * 100:.1f}% correct",
            f"named words, {sp['conversations']} test conversations", align=(0, 0), size=6.9)
    letter(cx, "c")
    save(fig, "fig4_corpus")


if __name__ == "__main__":
    fig_accuracy()
    fig_artifacts()
    fig_corpus()
    print("figures written to", OUT)
