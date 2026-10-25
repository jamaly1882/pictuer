#!/usr/bin/env python3
"""
Animated dialogue scene: Nour and her grandfather Salem talking at the well.

Uses the same stills as the film (film/assets) with slow Ken Burns camera moves,
cross-dissolves, English subtitles burned in, and the per-line narration/dialogue
clips in film/dialogue-v2/. Output: film/dialogue-v2/dialogue-animated.mp4

Reuses the rendering helpers from build_film.py (same folder).
"""
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_film as bf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SR = 1920, 1080, 25, 44100
LEAD, TAIL, XF = 0.6, 1.0, 0.6

# (audio file, still, English subtitle)
LINES = [
    ("l01-narration.mp3", "s01", "At dawn, Nour sat by the well, and her grandfather Salem found her quietly watching the water."),
    ("l02-nour.mp3", "s01", "Grandpa, why doesn't the water rise in the well the way it did when you were young? Every morning I count the knots, and the rope reaches deeper than yesterday."),
    ("l03-salem.mp3", "s02", "My child, water is like a trust. We let it seep into the earth through the terraces, and today we dig without limits."),
    ("l04-nour.mp3", "s02", "Is there still time to fix it?"),
    ("l05-salem.mp3", "s07", "There is still time, but we must start with what is in our hands. We rebuild the dams, and stop the water on the terraces before the flood sweeps it away."),
    ("l06-narration.mp3", "s10", "Nour smiled, picked up the blue notebook, and wrote the first line of the neighborhood plan."),
    ("l07-nour.mp3", "s10", "I will write everything down, so whoever comes after us knows how the water returned."),
    ("l08-salem.mp3", "s09", "This is the way, Nour. Write, plant, and wait for the rain gently."),
]


def main():
    out_mp4 = os.path.join(HERE, "dialogue-v2", "dialogue-animated.mp4")
    out_wav = "/tmp/dialogue-animated-mix.wav"
    font_sub = bf.load_font(bf.FONT_REG, 40)

    segs, cursor = [], 0.0
    for i, (aud, still, eng) in enumerate(LINES):
        audio = bf.decode_mono(os.path.join(HERE, "dialogue-v2", aud))
        d = len(audio) / SR
        img = Image.open(os.path.join(HERE, "assets", f"{still}.jpg")).convert("RGB")
        a_start = cursor + LEAD
        dur = LEAD + d + TAIL
        lines = [(a_start, a_start + d, eng)]
        segs.append(dict(start=cursor, dur=dur, base=img.resize((2048, 1152), Image.LANCZOS),
                         dirx=1 if i % 2 == 0 else -1, audio=audio, a_start=a_start, lines=lines))
        cursor += dur
    total = cursor
    print(f"total duration: {total:.2f}s", flush=True)
    starts = [s["start"] for s in segs]

    # ---- audio: dialogue + soft pad ----
    N = int(round(total * SR))
    narr = np.zeros(N, dtype=np.float32)
    for s in segs:
        a0 = int(round(s["a_start"] * SR))
        narr[a0:a0 + len(s["audio"])] = s["audio"]
    tt = np.arange(N, dtype=np.float32) / SR
    pad = np.zeros(N, dtype=np.float32)
    for k, f in enumerate([110.0, 130.81, 164.81, 220.0]):
        pad += np.sin(2 * np.pi * f * tt + k * 0.7) * (0.6 if k else 1.0)
    pad = pad * (0.75 + 0.25 * np.sin(2 * np.pi * tt / 11.0)) * 0.025
    pad = (pad * np.clip(tt / 2.0, 0, 1) * np.clip((total - tt) / 4.0, 0, 1)).astype(np.float32)
    mix = np.clip(narr + pad, -0.98, 0.98)
    with wave.open(out_wav, "wb") as wv:
        wv.setnchannels(1)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((mix * 32767).astype(np.int16).tobytes())

    sub_cache = {}

    def sub_for(text):
        if text not in sub_cache:
            sub_cache[text] = bf.make_sub_overlay(text, font_sub)
        return sub_cache[text]

    def render(seg, local):
        p = min(max(local / seg["dur"], 0.0), 1.0)
        z = 1.0 + 0.07 * p
        cw, ch = 2048 / z, 1152 / z
        x0 = (2048 - cw) * (0.5 + 0.4 * (p - 0.5) * seg["dirx"])
        y0 = (1152 - ch) * (0.5 - 0.15 * (p - 0.5))
        frame = seg["base"].resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        t_abs = seg["start"] + local
        for t0, t1, txt in seg["lines"]:
            if t0 <= t_abs <= t1:
                rgb, alpha = sub_for(txt)
                frame.paste(rgb, ((W - rgb.width) // 2, H - rgb.height - 90), alpha)
                break
        return frame

    ffcmd = [
        FF, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", out_wav, "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "28",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
        out_mp4,
    ]
    proc = subprocess.Popen(ffcmd, stdin=subprocess.PIPE)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    n_frames = int(round(total * FPS))
    idx = 0
    for f in range(n_frames):
        t = f / FPS
        while idx + 1 < len(segs) and t >= starts[idx + 1]:
            idx += 1
        seg = segs[idx]
        local = t - seg["start"]
        frame = render(seg, local)
        if local < XF and idx > 0:
            prev = segs[idx - 1]
            frame = Image.blend(render(prev, prev["dur"] - 1e-3), frame, local / XF)
        if t < 0.8:
            frame = Image.blend(black, frame, t / 0.8)
        if t > total - 1.2:
            frame = Image.blend(frame, black, (t - (total - 1.2)) / 1.2)
        proc.stdin.write(frame.tobytes())
        if f % (FPS * 10) == 0:
            print(f"frame {f}/{n_frames}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"done: {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
