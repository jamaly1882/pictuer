#!/usr/bin/env python3
"""
Builds "The Water Notebook" (دفتر الماء) from the generated stills and narration.

Pipeline
  - stills in film/assets/sNN.jpg (AI-generated), narration in film/audio/sNN.mp3 (AI voice)
  - slow Ken Burns camera moves + cross-dissolves, English subtitles burned in
  - soft ambient pad underneath the narration
  - output: film/output/the-water-notebook.mp4 (1920x1080, 25 fps, H.264 + AAC)
  - also writes film/subtitles-en.srt

Run with the venv that has pillow, numpy and imageio-ffmpeg installed.
"""
import bisect
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 25
SR = 44100
LEAD = 2.5      # silent lead-in before narration in each scene (s)
TAIL = 3.5      # silent hold after narration in each scene (s)
XF = 0.9        # cross-dissolve length (s)
TITLE_D = 8.0
END_D = 12.0
SUB_MAX_W = 1700

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SCENES = [
    ("s01", [
        "Back in Sana'a, on a cold dawn.",
        "Nour lowers an old rope into her grandfather's well.",
        "She counts the knots, one by one.",
        "Each knot is a meter of darkness. Every morning, the water has moved farther away.",
    ]),
    ("s02", [
        "Her grandfather Salem keeps a small notebook with a faded blue cover.",
        "Since he was young, he has written down the water level and every season's rain.",
        "\"When I was your age, the water was close to the surface,\" he says.",
        "\"Now we go down to depths no one imagined.\"",
    ]),
    ("s03", [
        "On the next street, a giant drill digs a new well, then another.",
        "Every neighbor digs deeper than the last, afraid the water will run out first.",
        "The Sana'a basin has more than thirteen thousand wells.",
        "They pump from an underground reservoir that is draining faster than it refills.",
    ]),
    ("s04", [
        "Nour stands with her mother and neighbors beside the water truck.",
        "The tank on their roof has been empty for weeks.",
        "Buying water by the tanker now strains the family budget.",
        "Her mother counts every liter. \"Water is counted now like gold,\" she says.",
    ]),
    ("s05", [
        "At school, Nour opens an AI app on her tablet and enters the notebook's measurements.",
        "The AI draws a curve that falls year after year.",
        "It shows that pumping far exceeds what the rain can refill.",
        "But a number alone cannot fill a well.",
        "What the notebook holds, no screen remembers: old springs, and the terraces her grandparents knew.",
    ]),
    ("s06", [
        "Then the rain comes all at once.",
        "Floodwater rushes down from the mountain, tearing through roads and walls.",
        "It passes through the neighborhood and never sinks into the ground.",
        "From the roof, Nour watches the water rush away to the valley,",
        "as if the earth had forgotten how to drink.",
    ]),
    ("s07", [
        "Nour and her grandfather walk down to the valley.",
        "They find an old stone dam, broken, and terraces left untended.",
        "\"Our grandparents held the water on the terraces,\" Salem says.",
        "\"It sank slowly into the earth and filled the wells. Today we let it pass.\"",
    ]),
    ("s08", [
        "The neighbors gather under an old tree.",
        "Nour shows a map that AI helped create from valley photos and the memories of elders.",
        "The young men propose rebuilding the stone dams, digging rainwater cisterns, and planting trees on the terraces.",
        "An elderly neighbor says, \"We cannot stop the drilling alone, but we can start with the water in our hands.\"",
    ]),
    ("s09", [
        "Months of shared work pass.",
        "This time the rain returns gently.",
        "The water soaks slowly into the soil and gathers in the new cisterns.",
        "At dawn, Nour lowers the rope and counts the knots.",
        "She stops at a knot higher than the one before.",
    ]),
    ("s10", [
        "Nour writes in the notebook in small letters: \"Today the water rose one knot. We are no longer alone.\"",
        "Salem smiles. \"Write down what we did,",
        "so whoever comes after us knows that water returns when we learn to listen.\"",
        "The sun sets behind the mountains of Sana'a, and the rope stretches down toward the water.",
    ]),
]

END_TEXT = [
    "Sana'a's aquifers are pumped at roughly twice their natural recharge.",
    "Harvesting rain and restoring terraces can help give the water back its time.",
]
END_CREDIT = "Story, narration and imagery produced with AI tools. Context: UNDP Yemen, Oxfam, Carnegie Middle East Center."
TITLE = "THE WATER NOTEBOOK"
TITLE_SUB = "A short film about groundwater, floods and community action in Sana'a, Yemen"


def decode_mono(path):
    raw = subprocess.run(
        [FF, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_font(path, size):
    return ImageFont.truetype(path, size)


def wrap_text(text, font, max_w):
    """Greedy word-wrap so every line fits in max_w pixels."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if not cur or font.getlength(test) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_sub_overlay(text, font):
    """Return (rgb, alpha) arrays for a subtitle box at bottom-centre (wrapped)."""
    pad_x, pad_y = 26, 14
    lines = wrap_text(text, font, SUB_MAX_W - 2 * pad_x)
    asc, desc = font.getmetrics()
    lh = asc + desc + 8
    tw = max(font.getlength(l) for l in lines)
    bw = int(min(tw + 2 * pad_x, SUB_MAX_W))
    bh = int(lh * len(lines) + 2 * pad_y)
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 150))
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        lw = font.getlength(l)
        d.text(((bw - lw) / 2, pad_y + i * lh), l, font=font, fill=(255, 255, 255, 255))
    arr = np.array(img)
    return Image.fromarray(arr[:, :, :3]), Image.fromarray(arr[:, :, 3])


def make_text_card(lines_fonts, dim=0.45):
    """Transparent overlay with centred text lines; lines_fonts = [(text, font, y, fill)]."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for text, font, y, fill in lines_fonts:
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        d.text(((W - (r - l)) // 2 - l, y - t), text, font=font, fill=fill)
    return ov


def main():
    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, "the-water-notebook.mp4")
    out_srt = os.path.join(HERE, "subtitles-en.srt")
    out_wav = os.path.join("/tmp", "the-water-notebook-mix.wav")

    font_sub = load_font(FONT_REG, 40)
    font_title = load_font(FONT_BOLD, 112)
    font_tsub = load_font(FONT_REG, 34)
    font_small = load_font(FONT_REG, 30)
    font_end = load_font(FONT_REG, 42)

    # ---------- timeline ----------
    segs = []  # dict: kind, start, dur, img, lines[(t0,t1,text)], audio, a_start
    cursor = 0.0

    title_img = Image.open(os.path.join(HERE, "assets", "s01.jpg")).convert("RGB")
    segs.append(dict(kind="title", start=cursor, dur=TITLE_D, img=title_img,
                     lines=[], audio=None, a_start=None))
    cursor += TITLE_D

    for sid, eng in SCENES:
        audio = decode_mono(os.path.join(HERE, "audio", f"{sid}.mp3"))
        d = len(audio) / SR
        img = Image.open(os.path.join(HERE, "assets", f"{sid}.jpg")).convert("RGB")
        a_start = cursor + LEAD
        total_chars = sum(len(x) for x in eng)
        lines, t = [], a_start
        for txt in eng:
            span = d * len(txt) / total_chars
            lines.append((t, t + span, txt))
            t += span
        dur = LEAD + d + TAIL
        segs.append(dict(kind="scene", sid=sid, start=cursor, dur=dur, img=img,
                         lines=lines, audio=audio, a_start=a_start))
        cursor += dur

    end_img = ImageEnhance.Brightness(Image.open(os.path.join(HERE, "assets", "s10.jpg")).convert("RGB")).enhance(0.35)
    segs.append(dict(kind="end", start=cursor, dur=END_D, img=end_img,
                     lines=[], audio=None, a_start=None))
    cursor += END_D
    total = cursor
    print(f"total duration: {total:.2f}s ({total/60:.2f} min)", flush=True)

    starts = [s["start"] for s in segs]

    # ---------- prepare per-segment assets ----------
    for s in segs:
        # upscale source to 2048x1152 for Ken Burns crops
        s["base"] = s["img"].resize((2048, 1152), Image.LANCZOS)
        s["idx_dir"] = 1 if segs.index(s) % 2 == 0 else -1

    title_overlay = make_text_card([
        (TITLE, font_title, H // 2 - 150, (255, 255, 255, 255)),
        (TITLE_SUB, font_tsub, H // 2 + 40, (235, 235, 235, 255)),
    ])
    title_ov_rgb = title_overlay.convert("RGB")
    title_ov_alpha = title_overlay.split()[3]
    end_lines = wrap_text(END_TEXT[0] + " " + END_TEXT[1], font_end, W - 360)
    end_entries = [(ln, font_end, H // 2 - 130 + i * 62, (255, 255, 255, 255)) for i, ln in enumerate(end_lines)]
    end_entries.append((END_CREDIT, font_small, H - 120, (200, 200, 200, 255)))
    end_overlay = make_text_card(end_entries)
    end_ov_rgb = end_overlay.convert("RGB")
    end_ov_alpha = end_overlay.split()[3]

    sub_cache = {}

    def sub_for(text):
        if text not in sub_cache:
            sub_cache[text] = make_sub_overlay(text, font_sub)
        return sub_cache[text]

    def render(seg, local):
        """Render the segment at local time (seconds) -> RGB PIL image."""
        p = min(max(local / seg["dur"], 0.0), 1.0)
        z = 1.0 + 0.07 * p if seg["idx_dir"] > 0 else 1.07 - 0.07 * p
        cw, ch = 2048 / z, 1152 / z
        dirx = seg["idx_dir"]
        x0 = (2048 - cw) * (0.5 + 0.4 * (p - 0.5) * dirx)
        y0 = (1152 - ch) * (0.5 - 0.15 * (p - 0.5))
        frame = seg["base"].resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        if seg["kind"] == "title":
            dark = Image.blend(frame, Image.new("RGB", (W, H), (0, 0, 0)), 0.45)
            frame = Image.composite(title_ov_rgb, dark, title_ov_alpha)
        elif seg["kind"] == "end":
            dark = Image.blend(frame, Image.new("RGB", (W, H), (0, 0, 0)), 0.3)
            frame = Image.composite(end_ov_rgb, dark, end_ov_alpha)
        else:
            for t0, t1, txt in seg["lines"]:
                if t0 <= seg["start"] + local <= t1:
                    rgb, alpha = sub_for(txt)
                    x = (W - rgb.width) // 2
                    y = H - rgb.height - 90
                    frame.paste(rgb, (x, y), alpha)
                    break
        return frame

    # ---------- audio mix ----------
    N = int(round(total * SR))
    narr = np.zeros(N, dtype=np.float32)
    for s in segs:
        if s.get("audio") is not None:
            a0 = int(round(s["a_start"] * SR))
            narr[a0:a0 + len(s["audio"])] = s["audio"]

    tt = np.arange(N, dtype=np.float32) / SR
    chord = [110.0, 130.81, 164.81, 220.0, 246.94]
    pad = np.zeros(N, dtype=np.float32)
    for i, f in enumerate(chord):
        pad += np.sin(2 * np.pi * f * tt + i * 0.7) * (0.6 if i else 1.0)
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * tt / 11.0)
    pad = pad * lfo * 0.03
    env = np.clip(tt / 3.0, 0, 1) * np.clip((total - tt) / 6.0, 0, 1)
    pad = (pad * env).astype(np.float32)
    mix = np.clip(narr * 1.0 + pad, -0.98, 0.98)
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes(pcm.tobytes())
    print("audio mixed", flush=True)

    # ---------- subtitles (SRT) ----------
    def ts(sec):
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s_, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s_:02d},{ms:03d}"

    n = 0
    with open(out_srt, "w", encoding="utf-8") as f:
        for s in segs:
            for t0, t1, txt in s.get("lines", []):
                n += 1
                f.write(f"{n}\n{ts(t0)} --> {ts(t1)}\n{txt}\n\n")
    print(f"srt written ({n} cues)", flush=True)

    # ---------- render video ----------
    ffcmd = [
        FF, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", out_wav,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
        out_mp4,
    ]
    proc = subprocess.Popen(ffcmd, stdin=subprocess.PIPE)
    n_frames = int(round(total * FPS))
    black = Image.new("RGB", (W, H), (0, 0, 0))
    seg_idx = 0
    for f in range(n_frames):
        t = f / FPS
        while seg_idx + 1 < len(segs) and t >= starts[seg_idx + 1]:
            seg_idx += 1
        seg = segs[seg_idx]
        local = t - seg["start"]
        frame = render(seg, local)
        # cross-dissolve from previous segment at its start
        if local < XF and seg_idx > 0:
            prev = segs[seg_idx - 1]
            prev_frame = render(prev, prev["dur"] - 1e-3)
            frame = Image.blend(prev_frame, frame, local / XF)
        # fade in from black / out to black
        if t < 1.0:
            frame = Image.blend(black, frame, t)
        if t > total - 1.5:
            frame = Image.blend(frame, black, (t - (total - 1.5)) / 1.5)
        proc.stdin.write(frame.tobytes())
        if f % (FPS * 20) == 0:
            print(f"frame {f}/{n_frames} ({100*f/n_frames:.0f}%)", flush=True)
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg failed with code {rc}")
    print(f"done: {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
