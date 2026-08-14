#!/usr/bin/env python3
"""
명언자동실행.py
영어 동기부여/명언 YouTube Shorts 자동 생성 + 업로드 스크립트

필요 패키지:
  pip install openai edge-tts moviepy google-api-python-client google-auth-oauthlib google-auth-httplib2 pillow

환경변수 / Secrets:
  XAI_API_KEY
  CLIENT_SECRETS
  TOKEN_PICKLE_B64
"""

import os
import sys
import json
import base64
import pickle
import random
import asyncio
import tempfile
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from openai import OpenAI
import edge_tts
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    ColorClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# ====================== 경로 ======================
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"
BACKGROUND_DIR = BASE_DIR / "backgrounds"
MUSIC_DIR = BASE_DIR / "music"
STATE_FILE = BASE_DIR / "last_used.json"   # 직전 배경/음악/저자 기록

OUTPUT_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)

BGM_VOLUME = 0.55


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"bg": None, "music": None, "quotes": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_random_exclude(files: list, last_name: str | None):
    """직전 파일은 제외하고 랜덤 선택. 파일이 1개면 그대로 사용."""
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    candidates = [f for f in files if f.name != last_name]
    if not candidates:
        candidates = files
    return random.choice(candidates)


def get_font_path() -> str:
    candidates = [
        BASE_DIR / "fonts" / "Montserrat-Bold.ttf",
        BASE_DIR / "fonts" / "Montserrat-SemiBold.ttf",
        BASE_DIR / "fonts" / "arial.ttf",
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] 사용 폰트: {path.name}")
            return str(path)
    print("[WARN] 폰트를 찾지 못했습니다. 기본 폰트 사용.")
    return "Arial"


FONT_PATH = get_font_path()

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    print("[ERROR] XAI_API_KEY 환경변수가 없습니다.")
    sys.exit(1)

VOICE = "en-US-AndrewNeural"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")


def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def wrap_by_pixel(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = text.split()
    if not words:
        return [text]
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        bbox = font.getbbox(test)
        if current and (bbox[2] - bbox[0]) > max_width:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def render_text_clip(txt: str, fontsize: int, start: float, end: float, y_pos="center"):
    """
    PIL 텍스트 렌더.
    글자 아래(descent) + 외곽선이 절대 잘리지 않도록
    실제 그려진 픽셀 bbox를 측정한 뒤 여백을 크게 둔다.
    """
    from moviepy import ImageClip

    max_width = 960
    fs = fontsize
    if len(txt) > 100:
        fs = max(40, fontsize - 14)
    elif len(txt) > 70:
        fs = max(46, fontsize - 8)

    try:
        font = ImageFont.truetype(FONT_PATH, fs)
    except Exception:
        font = ImageFont.load_default()

    lines = wrap_by_pixel(txt, font, max_width) or [txt]
    stroke = 5

    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = fs, max(10, fs // 3)

    line_gap = 28
    line_height = ascent + abs(descent) + line_gap
    # 여백을 매우 크게 (잘림 완전 차단)
    pad_top = 70
    pad_bottom = 80

    img_w = 1080
    img_h = pad_top + line_height * len(lines) + pad_bottom

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        tw = bbox[2] - bbox[0]
        x = (img_w - tw) // 2
        y = pad_top + i * line_height
        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    # 실제 그려진 영역 기준으로 한 번 더 여유 확인 (투명 여백만 남김)
    arr = np.array(img)
    clip = (
        ImageClip(arr, is_mask=False)
        .with_duration(max(0.7, end - start))
        .with_start(start)
    )

    if y_pos == "center":
        clip = clip.with_position(("center", "center"))
    else:
        if isinstance(y_pos, (int, float)):
            y_pos = min(int(y_pos), 1920 - img_h - 30)
        clip = clip.with_position(("center", y_pos))
    return clip


# ====================== 콘텐츠 생성 ======================
def generate_motivation_content(exclude_quotes: list) -> dict:
    """최근 사용 명언 문장과 겹치지 않게 새 콘텐츠 생성. 저자는 같아도 OK."""
    # 최근 명언 앞부분만 넣어서 프롬프트 길이 제한
    recent = []
    for q in exclude_quotes[-10:]:
        q = (q or "").strip()
        if q:
            recent.append(q[:80] + ("..." if len(q) > 80 else ""))
    exclude_txt = "\n".join(f'- "{r}"' for r in recent) if recent else "- (none yet)"

    prompt = f"""You are a professional motivational content creator specializing in YouTube Shorts.

Generate ONE high-quality daily motivation short in English.

CRITICAL — DO NOT REPEAT THESE RECENT QUOTES (same meaning or same wording):
{exclude_txt}

Rules:
- The quote itself must be DIFFERENT from the list above (new wording and new idea)
- Author may be anyone, including previously used authors — that is fine
- Prefer a powerful, relatively non-cliché quote
- Insightful commentary MUST be 3-4 full sentences (not short), calm / wise / premium tone
- Total spoken content (hook + quote + commentary) MUST be 95-120 English words so the video lasts 30-35 seconds at slow speech
- Prefer longer, reflective sentences over short slogans

Output ONLY valid JSON (no markdown):

{{
  "quote": "the full quote here",
  "author": "Author Name or null",
  "commentary": "3-4 full sentences of insightful commentary, ~60-80 words",
  "title": "attractive Shorts title under 70 characters",
  "description": "YouTube description ending with 5-7 relevant hashtags",
  "hook": "strong 4-8 word hook text for the first 2 seconds"
}}"""

    print("[INFO] Grok으로 명언 콘텐츠 생성 중...")
    if recent:
        print(f"[INFO] 최근 명언 {len(recent)}개 제외 요청")
    response = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[ERROR] JSON 파싱 실패:")
        print(raw)
        raise e
    print(f"[OK] Quote: {data['quote'][:60]}...")
    print(f"[OK] Author: {data.get('author')}")
    return data


# ====================== TTS + BGM ======================
async def _generate_tts(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, VOICE, rate="-15%")
    await communicate.save(output_path)


def make_audio(hook: str, quote: str, author, commentary: str, last_music: str | None) -> tuple:
    parts = []
    if hook:
        parts.append(hook.strip().rstrip(".") + ".")
    if quote:
        parts.append(quote.strip().rstrip(".") + ".")
    if author and str(author).lower() not in ("null", "none", ""):
        parts.append(str(author).strip() + ".")
    if commentary:
        parts.append(commentary.strip())

    full_text = " ".join(parts)
    print(f"[INFO] TTS 텍스트: {full_text[:80]}...")

    tts_path = OUTPUT_DIR / "tts.mp3"
    print("[INFO] TTS 생성 중...")
    asyncio.run(_generate_tts(full_text, str(tts_path)))
    print(f"[OK] TTS 저장: {tts_path.name}")

    audio_path, used_music = mix_with_bgm(str(tts_path), last_music)
    return audio_path, used_music


def mix_with_bgm(tts_path: str, last_music: str | None) -> tuple:
    import subprocess
    import shutil

    seen = set()
    music_files = []
    for pattern in ("*.mp3", "*.wav", "*.m4a", "*.mp4", "*.MP3", "*.MP4"):
        for f in MUSIC_DIR.glob(pattern):
            key = f.name.lower()
            if key not in seen:
                seen.add(key)
                music_files.append(f)

    print(f"[INFO] music 파일 수: {len(music_files)}")
    for f in music_files:
        print(f"       - {f.name}")

    if not music_files:
        print("[WARN] music 폴더 비어 있음 → TTS만 사용")
        return tts_path, None

    bgm_path = pick_random_exclude(music_files, last_music)
    print(f"[INFO] 선택된 배경음악: {bgm_path.name} (직전: {last_music})")

    output_path = OUTPUT_DIR / "audio.mp3"
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        print("[WARN] ffmpeg 없음 → MoviePy 시도")
        return _mix_with_moviepy(tts_path, bgm_path, output_path), bgm_path.name

    try:
        tts_clip = AudioFileClip(tts_path)
        tts_dur = tts_clip.duration
        tts_clip.close()
        target = tts_dur + 1.5

        cmd = [
            ffmpeg, "-y",
            "-i", tts_path,
            "-i", str(bgm_path),
            "-filter_complex",
            f"[1:a]volume={BGM_VOLUME},atrim=0:{target},asetpts=PTS-STARTPTS[bg];"
            f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[voice];"
            f"[bg]aformat=sample_rates=44100:channel_layouts=stereo[bg2];"
            f"[voice][bg2]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map", "[out]",
            "-t", str(target),
            "-ac", "2",
            "-ar", "44100",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            print("[WARN] ffmpeg 실패 → MoviePy")
            return _mix_with_moviepy(tts_path, bgm_path, output_path), bgm_path.name

        print(f"[OK] ffmpeg 믹싱 완료: {output_path.name}")
        return str(output_path), bgm_path.name
    except Exception as e:
        print(f"[ERROR] BGM 믹싱 실패: {e}")
        return tts_path, bgm_path.name


def _mix_with_moviepy(tts_path: str, bgm_path: Path, output_path: Path) -> str:
    try:
        tts = AudioFileClip(tts_path)
        video_clip = None
        if bgm_path.suffix.lower() == ".mp4":
            video_clip = VideoFileClip(str(bgm_path))
            bgm = video_clip.audio
            if bgm is None:
                if video_clip:
                    video_clip.close()
                return tts_path
        else:
            bgm = AudioFileClip(str(bgm_path))

        target = tts.duration + 1.5
        bgm = bgm.subclipped(0, min(bgm.duration, target))
        try:
            bgm = bgm.with_volume_scaled(BGM_VOLUME)
        except Exception:
            bgm = bgm.volumex(BGM_VOLUME)

        final = CompositeAudioClip([bgm, tts]).with_duration(target)
        final.write_audiofile(str(output_path), fps=44100, codec="libmp3lame", logger=None)
        tts.close(); bgm.close(); final.close()
        if video_clip:
            video_clip.close()
        print(f"[OK] MoviePy 믹싱 완료")
        return str(output_path)
    except Exception as e:
        print(f"[ERROR] MoviePy 믹싱 실패: {e}")
        return tts_path


# ====================== 영상 ======================
def make_background(bg_path: Path, duration: float):
    import subprocess
    import shutil

    bg = VideoFileClip(str(bg_path)).without_audio()
    bg = bg.resized(height=1920)
    if bg.w > 1080:
        bg = bg.cropped(x_center=bg.w / 2, width=1080, height=1920)

    tmp_dir = Path(tempfile.mkdtemp())
    fwd_path = tmp_dir / "fwd.mp4"
    rev_path = tmp_dir / "rev.mp4"
    bg.write_videofile(str(fwd_path), fps=30, codec="libx264", audio=False, logger=None)

    bg_rev = None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(fwd_path), "-vf", "reverse", "-an", str(rev_path)],
                check=True, capture_output=True, encoding='utf-8', errors='replace',
            )
            bg_rev = VideoFileClip(str(rev_path)).without_audio()
            print("[INFO] 배경 핑퐁 ON")
        except Exception as e:
            print(f"[WARN] 역재생 실패: {e}")
    else:
        print("[WARN] ffmpeg 없음 → 일반 반복")

    clips, t, forward = [], 0.0, True
    while t < duration + 0.5:
        clips.append(bg if (forward or bg_rev is None) else bg_rev)
        t += bg.duration
        forward = not forward

    return concatenate_videoclips(clips).subclipped(0, duration).with_fps(30)


def create_video(content: dict, audio_path: str, last_bg: str | None) -> tuple:
    print("[INFO] 영상 합성 중...")
    audio = AudioFileClip(audio_path)
    duration = audio.duration + 1.8

    bg_files = list(BACKGROUND_DIR.glob("*.mp4")) + list(BACKGROUND_DIR.glob("*.mov"))
    print(f"[INFO] backgrounds 파일 수: {len(bg_files)}")
    for f in bg_files:
        print(f"       - {f.name}")

    used_bg = None
    if not bg_files:
        background = ColorClip(size=(1080, 1920), color=(15, 15, 25)).with_duration(duration)
    else:
        bg_path = pick_random_exclude(bg_files, last_bg)
        used_bg = bg_path.name
        print(f"[INFO] 선택된 배경: {used_bg} (직전: {last_bg})")
        background = make_background(bg_path, duration)

    def make_text(txt, fontsize, start, end, y_pos="center"):
        return render_text_clip(txt, fontsize, start, end, y_pos=y_pos)

    hook = (content.get("hook") or "").strip()
    quote = (content.get("quote") or "").strip()
    author = content.get("author")
    if not author or str(author).lower() in ("null", "none", ""):
        author = "Unknown"
    commentary = (content.get("commentary") or "").strip()

    quote_sents = split_sentences(quote) or ([quote] if quote else [])
    comment_sents = split_sentences(commentary)

    main_sents, kinds = [], []
    if hook:
        main_sents.append(hook)
        kinds.append("Hook")
    for s in quote_sents:
        main_sents.append(s)
        kinds.append("명언")
    for s in comment_sents:
        main_sents.append(s)
        kinds.append("코멘트")

    speak_dur = max(1.0, audio.duration - 0.4)
    weights = [max(len(s), 10) for s in main_sents] or [1]
    total_w = sum(weights)
    times, t = [], 0.0
    for w in weights:
        seg = speak_dur * (w / total_w)
        times.append((t, t + seg))
        t += seg

    layers = [background]

    # 저자: 영상 처음부터 끝까지 하단 고정
    if author:
        layers.append(make_text(f"— {author}", 40, 0, duration, y_pos=1400))
        print(f"   저자 전체구간: 0~{duration:.1f}s | — {author}")

    for i, sent in enumerate(main_sents):
        start_t, end_t = times[i]
        fs = 68 if kinds[i] == "Hook" else (58 if kinds[i] == "명언" else 50)
        layers.append(make_text(sent, fs, start_t, end_t, y_pos="center"))
        print(f"   {kinds[i]}{i+1}: {start_t:.1f}~{end_t:.1f}s | {sent[:55]}...")

    final = CompositeVideoClip(layers, size=(1080, 1920)).with_audio(audio)
    output_path = OUTPUT_DIR / "motivation.mp4"
    final.write_videofile(
        str(output_path), fps=30, codec="libx264", audio_codec="aac",
        threads=4, preset="medium", logger=None,
    )
    audio.close(); background.close(); final.close()
    print(f"[OK] 영상 생성 완료: {output_path.name}")
    return str(output_path), used_bg


# ====================== 유튜브 ======================
def get_authenticated_service():
    """
    인증 우선순위:
      1) 환경변수 TOKEN_PICKLE_B64 / CLIENT_SECRETS  (GitHub Actions)
      2) 로컬 파일 token.pickle / client_secrets.json  (Windows 로컬)
    """
    creds = None
    token_path = BASE_DIR / "token.pickle"
    secrets_path = BASE_DIR / "client_secrets.json"

    # 1) 환경변수 토큰
    token_b64 = os.getenv("TOKEN_PICKLE_B64")
    if token_b64:
        try:
            creds = pickle.loads(base64.b64decode(token_b64))
            print("[INFO] 인증: TOKEN_PICKLE_B64 환경변수 사용")
        except Exception as e:
            print(f"[WARN] TOKEN_PICKLE_B64 복원 실패: {e}")

    # 2) 로컬 token.pickle
    if creds is None and token_path.exists():
        try:
            with open(token_path, "rb") as f:
                creds = pickle.load(f)
            print(f"[INFO] 인증: 로컬 {token_path.name} 사용")
        except Exception as e:
            print(f"[WARN] token.pickle 로드 실패: {e}")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] 토큰 갱신 중...")
            creds.refresh(Request())
            # 로컬이면 갱신된 토큰 저장
            try:
                with open(token_path, "wb") as f:
                    pickle.dump(creds, f)
            except Exception:
                pass
        else:
            # client_secrets 확보
            client_secrets_content = os.getenv("CLIENT_SECRETS")
            secrets_file = None
            if client_secrets_content:
                print("[INFO] 인증: CLIENT_SECRETS 환경변수 사용")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
                    f.write(client_secrets_content)
                    secrets_file = f.name
            elif secrets_path.exists():
                print(f"[INFO] 인증: 로컬 {secrets_path.name} 사용")
                secrets_file = str(secrets_path)
            else:
                raise RuntimeError(
                    "YouTube 인증 정보가 없습니다.\n"
                    "  - 로컬: client_secrets.json + token.pickle 을 스크립트 폴더에 두세요\n"
                    "  - GitHub: CLIENT_SECRETS, TOKEN_PICKLE_B64 Secrets 등록"
                )

            flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
            if secrets_file != str(secrets_path):
                try:
                    os.unlink(secrets_file)
                except Exception:
                    pass
            # 로컬 토큰 저장
            try:
                with open(token_path, "wb") as f:
                    pickle.dump(creds, f)
                print(f"[OK] 토큰 저장: {token_path.name}")
            except Exception as e:
                print(f"[WARN] token.pickle 저장 실패: {e}")

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: str, title: str, description: str) -> str:
    print("[INFO] 유튜브 업로드 중...")
    youtube = get_authenticated_service()
    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": [
                "motivation", "inspirational", "quotes", "daily motivation",
                "shorts", "english quotes", "self improvement", "mindset"
            ],
            "categoryId": "22",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
            "madeForKids": False,
        },
    }
    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"   업로드 진행률: {int(status.progress() * 100)}%")
    video_id = response.get("id")
    url = f"https://youtube.com/shorts/{video_id}"
    print(f"[OK] 업로드 완료: {url}")
    return url


# ====================== 메인 ======================
def main():
    print("=" * 50)
    print(" 영어 명언 Shorts 자동 생성 시작")
    print("=" * 50)

    state = load_state()
    print(f"[INFO] 직전 상태: bg={state.get('bg')}, music={state.get('music')}")
    print(f"[INFO] 최근 명언 수: {len(state.get('quotes', []))}")

    content = generate_motivation_content(state.get("quotes", []))

    audio_path, used_music = make_audio(
        content.get("hook", ""),
        content["quote"],
        content.get("author"),
        content["commentary"],
        state.get("music"),
    )

    video_path, used_bg = create_video(content, audio_path, state.get("bg"))

    # 상태 저장 (연속 배경/음악 중복 방지 + 최근 명언 기록)
    quotes = state.get("quotes", [])
    q = (content.get("quote") or "").strip()
    if q:
        quotes.append(q)
        quotes = quotes[-12:]  # 최근 12개 명언만 유지
    state = {"bg": used_bg, "music": used_music, "quotes": quotes}
    save_state(state)
    print(f"[INFO] 상태 저장: bg={used_bg}, music={used_music}, quotes={len(quotes)}개")

    url = upload_to_youtube(video_path, content["title"], content["description"])

    print("\n" + "=" * 50)
    print(" 모든 작업 완료!")
    print(f"제목 : {content['title']}")
    print(f"영상 : {video_path}")
    print(f"URL  : {url}")
    print("=" * 50)


if __name__ == "__main__":
    main()
