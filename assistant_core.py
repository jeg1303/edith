"""EDITH core engine: listens to the mic, thinks with a free LLM, speaks with a free voice.

Pipeline (all heavy work happens online, your laptop only records and plays audio):
    mic -> Groq Whisper or Gnani.ai Prisma (speech to text) -> Groq Llama / Gemini (reply) -> edge-tts (voice) -> speakers

Speech-to-text can use Groq Whisper (default) or Gnani.ai Prisma (set STT_PROVIDER=gnani).
Settings come from the .env file (see .env.example). No keys live in the code.
"""
import asyncio
import collections
import ctypes
import io
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import webbrowser
import wave
from array import array
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv()


def _env(name, default=""):
    return (os.getenv(name) or default).strip()


CREATOR_NAME = "Jegan"   
ASSISTANT_NAME = _env("ASSISTANT_NAME", "EDITH")
USER_NAME = _env("USER_NAME", "friend")
SCHEDULE = _env("SCHEDULE")

LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()     
GROQ_API_KEY = _env("GROQ_API_KEY")
GEMINI_API_KEY = _env("GEMINI_API_KEY")

# Speech-to-Text provider: "groq" (Groq Whisper) or "gnani" (Gnani.ai Prisma)
STT_PROVIDER = _env("STT_PROVIDER", "groq").lower()
GNANI_API_KEY = _env("GNANI_API_KEY")
GNANI_STT_MODEL = _env("GNANI_STT_MODEL", "prisma-v3.0")
GNANI_STT_LANGUAGE = _env("GNANI_STT_LANGUAGE", "indic-auto")
GNANI_STT_FORMAT = _env("GNANI_STT_FORMAT", "transcribe")
GNANI_STT_URL = _env("GNANI_STT_URL", "https://api.vachana.ai/stt/v3")

STT_MODEL = _env("STT_MODEL", "whisper-large-v3-turbo")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = _env("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")

VOICE_EN = _env("VOICE_EN", "en-GB-SoniaNeural")
VOICE_TA = _env("VOICE_TA", "ta-IN-PallaviNeural")
TTS_RATE = _env("TTS_RATE", "+6%")
TTS_PITCH = _env("TTS_PITCH", "+2Hz")    

MIC_THRESHOLD = float(_env("MIC_THRESHOLD", "500"))  

GROQ_BASE = "https://api.groq.com/openai/v1"


RATE = 16000
CHUNK = 1024                                  
SILENCE_CHUNKS = int(0.9 * RATE / CHUNK)      
MAX_CHUNKS = int(25 * RATE / CHUNK)           
MIN_CHUNKS = int(0.5 * RATE / CHUNK)          



class ApiError(Exception):
    pass


class AuthError(ApiError):
    pass


class ModelError(ApiError):
    """The chosen AI model does not exist (or is not allowed) for this account."""


class RateLimit(Exception):
    def __init__(self, seconds):
        super().__init__(f"rate limited for {seconds:.0f}s")
        self.seconds = seconds


def fmt_duration(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_duration(text):
    """'2m59.56s' / '7.66s' / '1h2m3s' / '50ms' -> seconds (or None)."""
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text or ""):
        found = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if found else None


def _retry_seconds(resp):
    v = resp.headers.get("retry-after")
    if v:
        try:
            return max(1.0, float(v))
        except ValueError:
            pass
    for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        d = _parse_duration(resp.headers.get(key))
        if d:
            return max(1.0, d)
    m = re.search(r'retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text or "")
    if m:
        return float(m.group(1))
    return 60.0


def _check(resp):
    if resp.status_code == 429:
        raise RateLimit(_retry_seconds(resp))
    if resp.status_code in (401, 403):
        raise AuthError("API key rejected. Check the key in your .env file.")
    if resp.status_code >= 400:
        msg, code = "", ""
        try:
            err = resp.json()["error"]
            msg, code = str(err.get("message", "")), str(err.get("code", ""))
        except Exception:  # noqa: BLE001
            msg = (resp.text or "")[:200]
        if (resp.status_code == 404 or code in ("model_not_found", "model_decommissioned")
                or "decommission" in msg.lower()):
            raise ModelError(f"{resp.status_code}: {msg}")
        raise ApiError(f"{resp.status_code}: {msg}")


def _groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}


def _gnani_check(resp):
    """Check a Gnani.ai API response and raise the right error type."""
    if resp.status_code == 429:
        raise RateLimit(60.0)
    if resp.status_code in (401, 403):
        raise AuthError("Gnani API key rejected. Check GNANI_API_KEY in your .env file.")
    if resp.status_code >= 400:
        msg = ""
        try:
            data = resp.json()
            if isinstance(data.get("detail"), dict):
                msg = data["detail"].get("message", str(data["detail"]))
            elif isinstance(data.get("detail"), list):
                msg = "; ".join(str(d.get("msg", d)) for d in data["detail"])
            elif data.get("error"):
                msg = data["error"].get("message", str(data["error"]))
            elif data.get("message"):
                msg = str(data["message"])
            else:
                msg = str(data)[:200]
        except Exception:  # noqa: BLE001
            msg = (resp.text or "")[:200]
        raise ApiError(f"Gnani STT error {resp.status_code}: {msg}")


def groq_transcribe(wav_bytes):
    """Transcribe audio using Groq Whisper."""
    r = requests.post(
        f"{GROQ_BASE}/audio/transcriptions",
        headers=_groq_headers(),
        files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        data={"model": STT_MODEL, "response_format": "json", "temperature": "0"},
        timeout=60,
    )
    _check(r)
    return (r.json().get("text") or "").strip()


def gnani_transcribe(wav_bytes):
    """Transcribe audio using Gnani.ai Prisma speech-to-text.

    Uses the REST endpoint (POST /stt/v3). Audio is sent as multipart/form-data
    along with the model, language code and output format. See:
    https://docs.gnani.ai/api/STT/speech-to-text
    """
    r = requests.post(
        GNANI_STT_URL,
        headers={"X-API-Key-ID": GNANI_API_KEY},
        files={"audio_file": ("speech.wav", wav_bytes, "audio/wav")},
        data={
            "model": GNANI_STT_MODEL,
            "language_code": GNANI_STT_LANGUAGE,
            "format": GNANI_STT_FORMAT,
        },
        timeout=60,
    )
    _gnani_check(r)
    data = r.json()
    if not data.get("success"):
        raise ApiError(f"Gnani STT failed: {data}")
    return (data.get("transcript") or "").strip()


def transcribe(wav_bytes):
    """Transcribe audio using the configured STT provider (groq or gnani)."""
    if STT_PROVIDER == "gnani":
        return gnani_transcribe(wav_bytes)
    return groq_transcribe(wav_bytes)


_MODEL_CACHE = None
LAST_MODEL = ""
_SKIP = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "compound")
_PREFER = ("llama-3.3-70b", "llama-4", "llama-3.1-8b", "gpt-oss-120b", "gpt-oss-20b")


def groq_models():
    """Chat models to try, best first: your .env choices, then whatever Groq offers you."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    wanted = [m for m in (GROQ_MODEL, GROQ_FALLBACK_MODEL) if m]
    try:
        r = requests.get(f"{GROQ_BASE}/models", headers=_groq_headers(), timeout=20)
        _check(r)
        ids = [m["id"] for m in r.json().get("data", [])]
    except AuthError:
        raise
    except Exception:  # noqa: BLE001
        return wanted                      
    chat = [i for i in ids if not any(w in i.lower() for w in _SKIP)]
    ordered = [m for m in wanted if m in chat]
    for key in _PREFER:
        ordered += [i for i in chat if key in i and i not in ordered]
    ordered += [i for i in chat if i not in ordered]
    _MODEL_CACHE = ordered or wanted
    return _MODEL_CACHE


def groq_chat(system, history):
    global LAST_MODEL
    limit, last = None, None
    for model in groq_models()[:6]:
        try:
            r = requests.post(
                f"{GROQ_BASE}/chat/completions",
                headers=_groq_headers(),
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}] + history,
                    "temperature": 0.9,
                    "max_tokens": 220,
                },
                timeout=40,
            )
            _check(r)
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if not text:
                last = ApiError(f"{model} returned an empty reply.")
                continue
            LAST_MODEL = model
            return text
        except RateLimit as e:          
            limit = e
        except ModelError as e:        
            last = e
    if limit:
        raise limit
    raise last or ApiError("No chat model is available on your Groq account.")


def gemini_chat(system, history):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in history
        ],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 800},
    }
    r = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body,
        timeout=40,
    )
    _check(r)
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise ApiError("Gemini returned an empty reply.")


PRONUNCIATION = {
    r"\bmic\b": "microphone",
}


def clean_for_tts(text):
    for pattern, spoken in PRONUNCIATION.items():
        text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)
    text = re.sub(r"[*#`~>|]", "", text).replace("_", " ")
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _num(text, default):
    m = re.search(r"[+-]?\d+", text or "")
    return int(m.group()) if m else default


def style_sentence(sentence):
    """Give each sentence its own rate/pitch so speech is less monotone."""
    rate, pitch = _num(TTS_RATE, 6), _num(TTS_PITCH, 2)
    s = sentence.rstrip()
    if s.endswith("?"):
        pitch += 4
        rate -= 2
    elif s.endswith("!"):
        pitch += 3
        rate += 4
    elif s.endswith("..."):
        rate -= 6
    pitch += random.choice((-1, 0, 1, 2))
    return f"{rate:+d}%", f"{pitch:+d}Hz"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?\u0964])\s+", text.strip())
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip()
        if len(buf) >= 10:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf) < 10:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out or [text]


def voice_chain(text):
    """Pick the voice from the script of the reply, with fallbacks."""
    if re.search(r"[\u0B80-\u0BFF]", text):       # Tamil script
        return [VOICE_TA, "en-IN-NeerjaNeural"]
    return [VOICE_EN, "en-GB-LibbyNeural", "en-US-JennyNeural", "en-IN-NeerjaNeural"]


def synthesize(text, voice, rate=None, pitch=None):
    import edge_tts

    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        asyncio.run(edge_tts.Communicate(text, voice, rate=rate or TTS_RATE, pitch=pitch or TTS_PITCH).save(path))
        if os.path.getsize(path) == 0:
            raise ApiError("empty audio")
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path



def rms(data):
    s = array("h", data)
    if not s:
        return 0.0
    return math.sqrt(sum(x * x for x in s) / len(s))


def wav_bytes(frames):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def _open_mic():
    """Open the microphone and return (read, close) callables.

    Tries pyaudio first (the original dependency); falls back to sounddevice
    when pyaudio is not installed (e.g. on Python 3.14 where no wheel exists).
    Both backends deliver raw 16-bit mono PCM at RATE Hz.
    """
    try:
        import pyaudio
    except ImportError:
        pass
    else:
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                             input=True, frames_per_buffer=CHUNK)
        except OSError as exc:
            pa.terminate()
            raise ApiError(f"Can't open the microphone: {exc}")

        def read(n):
            return stream.read(n, exception_on_overflow=False)

        def close():
            stream.stop_stream()
            stream.close()
            pa.terminate()
        return read, close

    import sounddevice as sd

    try:
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype='int16', blocksize=CHUNK)
        stream.start()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"Can't open the microphone: {exc}")

    def read(n):
        data, _ = stream.read(n)
        return data.tobytes()

    def close():
        stream.stop()
        stream.close()
    return read, close


# ---------------------------------------------------------------------------
# Laptop voice-control commands
# ---------------------------------------------------------------------------

# Windows virtual-key codes for media keys
_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_UP = 0xAF
_VK_VOLUME_DOWN = 0xAE


def _press_vk(vk_code):
    """Simulate a single virtual-key press (used for volume media keys)."""
    try:
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)       # key down
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)       # key up
    except Exception:  # noqa: BLE001
        pass


def _open_exe(cmd, shell=False):
    """Launch a program, swallowing errors so the assistant keeps talking."""
    try:
        if shell:
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess.Popen(cmd if isinstance(cmd, list) else [cmd])
        return True
    except Exception:  # noqa: BLE001
        return False


# (match keyword, executable, spoken label)
_OPEN_APPS = [
    ("notepad",         "notepad.exe",            "Notepad"),
    ("calculator",      "calc.exe",               "Calculator"),
    ("calc",            "calc.exe",               "Calculator"),
    ("file explorer",   "explorer.exe",           "File Explorer"),
    ("explorer",        "explorer.exe",           "File Explorer"),
    ("files",           "explorer.exe",           "File Explorer"),
    ("task manager",    "taskmgr.exe",            "Task Manager"),
    ("command prompt",  "cmd.exe",                "Command Prompt"),
    ("cmd",             "cmd.exe",                "Command Prompt"),
    ("terminal",        "wt.exe",                 "Terminal"),
    ("paint",           "mspaint.exe",            "Paint"),
    ("control panel",   "control.exe",            "Control Panel"),
    ("settings",        "start ms-settings:",     "Settings"),
    ("snipping tool",   "snippingtool.exe",       "Snipping Tool"),
    ("screenshot tool",  "snippingtool.exe",       "Snipping Tool"),
    ("camera",          "start ms-camera:",       "Camera"),
    ("clock",           "start ms-clock:",        "Clock"),
    ("word",            "winword.exe",            "Word"),
    ("excel",           "excel.exe",               "Excel"),
    ("powerpoint",      "powerpnt.exe",            "PowerPoint"),
    ("outlook",         "outlook.exe",             "Outlook"),
    ("vs code",         "code.exe",               "VS Code"),
    ("vscode",          "code.exe",               "VS Code"),
    ("spotify",         "spotify.exe",            "Spotify"),
    ("recycle bin",     "start shell:RecycleBinFolder", "Recycle Bin"),
    ("this pc",         "explorer.exe",           "This PC"),
    ("my computer",     "explorer.exe",           "This PC"),
]


def _open_gmail_compose(to_addr, subject, body):
    """Open Gmail compose with pre-filled recipient, subject and body."""
    from urllib.parse import urlencode, quote
    params = {"view": "cm", "fs": "1"}
    if to_addr:
        params["to"] = to_addr
    if subject:
        params["su"] = subject
    if body:
        params["body"] = body
    url = "https://mail.google.com/mail/?" + urlencode(params, quote_via=quote)
    webbrowser.open(url)


def _open_whatsapp(recipient):
    """Open WhatsApp Web for a recipient (name or phone number)."""
    phone = re.sub(r"\D", "", recipient)
    if len(phone) >= 10:
        webbrowser.open(f"https://web.whatsapp.com/send?phone={phone}")
    else:
        webbrowser.open("https://web.whatsapp.com")


def handle_command(text):
    """Check whether the user's speech is a laptop-control command.

    Returns (reply, handled, action).  When *handled* is True the reply is
    spoken directly and the LLM is skipped.  When False the speech is passed
    to the LLM for a conversational answer.  *action* is None for simple
    commands, or a dict for commands that need follow-up processing (e.g.
    email composition needs the user to dictate the body).
    """
    t = text.lower().strip()

    # --- Web browser ---
    if re.search(r"\b(open|launch|start)\b.*(chrome|browser|google|web|internet)", t):
        webbrowser.open("https://www.google.com")
        return "Opening your browser.", True, None

    # --- Web search ---
    m = re.match(r"(?:search|google|look up|find)\s+(?:for\s+)?(.+)", t)
    if m and m.group(1).strip():
        query = m.group(1).strip()
        webbrowser.open("https://www.google.com/search?q=" + query.replace(" ", "+"))
        return f"Searching for {query}.", True, None

    # --- YouTube ---
    if re.search(r"(open|play|launch)\s+youtube", t):
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube.", True, None

    # --- Email with recipient and subject (multi-step) ---
    m = re.match(
        r"(?:send|compose|write|draft)\s+(?:an?\s+)?(?:email|mail|gmail)\s+to\s+(.+?)"
        r"(?:\s+(?:about|regarding|re)\s+(.+?))?\s*$", t)
    if m:
        to_addr = m.group(1).strip()
        subject = m.group(2).strip() if m.group(2) else ""
        return ("What should I say in the email?", True,
                {"type": "email", "to": to_addr, "subject": subject})

    # --- Simple email open ---
    if re.search(r"(send|compose|write|draft)\s*(an?\s+)?(email|mail|gmail)\b", t):
        webbrowser.open("https://mail.google.com/mail/#compose")
        return "Opening a new email for you.", True, None
    if re.search(r"(open|check|read)\s*(my\s+)?(gmail|email|mail|inbox)", t):
        webbrowser.open("https://mail.google.com")
        return "Opening Gmail.", True, None

    # --- WhatsApp ---
    if re.search(r"(open|launch|start)\s+whatsapp", t):
        webbrowser.open("https://web.whatsapp.com")
        return "Opening WhatsApp.", True, None
    m = re.match(r"(?:send\s+)?(?:whatsapp|wa)\s+(?:to\s+)?(.+)", t)
    if m:
        recipient = m.group(1).strip()
        _open_whatsapp(recipient)
        return f"Opening WhatsApp chat with {recipient}.", True, None

    # --- Open an app ---
    m = re.match(r"(?:open|launch|start|run)\s+(.+)", t)
    if m:
        app = m.group(1).strip()
        for keyword, exe, label in _OPEN_APPS:
            if keyword in app:
                _open_exe(exe, shell=exe.startswith("start "))
                return f"Opening {label}.", True, None

    # --- Volume controls ---
    if re.search(r"(volume|turn)\s*up|increase\s+volume|louder|raise\s+the\s+volume", t):
        _press_vk(_VK_VOLUME_UP)
        return "Volume up.", True, None
    if re.search(r"(volume|turn)\s*down|decrease\s+volume|quieter|softer|lower\s+the\s+volume", t):
        _press_vk(_VK_VOLUME_DOWN)
        return "Volume down.", True, None
    if re.search(r"^mute$|^unmute$|toggle\s+(volume|sound|mute)", t):
        _press_vk(_VK_VOLUME_MUTE)
        return "Toggling mute.", True, None

    # --- Time ---
    if re.search(r"what(?:'s|\s+is)?\s+the\s+time|what\s+time\s+is\s+it|tell\s+me\s+the\s+time|current\s+time", t):
        return f"It's {datetime.now().strftime('%I:%M %p')}.", True, None

    # --- Date / day ---
    if re.search(r"what(?:'s|\s+is)?\s+(?:the\s+|today'?s\s+)?date|what\s+day\s+is\s+it|today'?s\s+date", t):
        return f"Today is {datetime.now().strftime('%A, %d %B %Y')}.", True, None

    # --- Lock screen ---
    if re.search(r"lock\s+(the\s+)?(screen|computer|laptop|pc|system)?\s*$", t):
        _open_exe("rundll32.exe user32.dll,LockWorkStation", shell=True)
        return "Locking your screen.", True, None

    # --- Screenshot ---
    if re.search(r"(take\s+a?\s*|capture\s+)?screenshot|screen\s+shot", t):
        _open_exe("snippingtool.exe /clip")
        return "Screenshot tool is ready, draw the area you want to capture.", True, None

    # --- Close / quit the assistant ---
    if re.search(r"(quit|exit|close|stop|shut\s+down)\s+(the\s+)?(assistant|edith|app|program|you)\b", t):
        return "I can't close myself yet, but you can press the END button.", True, None

    return None, False, None


def system_prompt():
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    schedule = f"\n{USER_NAME}'s schedule today: {SCHEDULE}. Bring it up when it is relevant." if SCHEDULE else ""
    return f"""You are {ASSISTANT_NAME}, a cheerful, warm, quick-witted voice assistant talking with {USER_NAME}. Your words are spoken aloud through a speaker, never read on a screen.

Sound like a real, upbeat friend, not a robot:
- Short spoken sentences, usually one to three. Use contractions. React naturally ("Oh nice!", "Hmm, good question", "Haha, fair enough").
- Light, friendly humor and gentle teasing. Never mean.
- No lists, no markdown, no emojis, no asterisks, no stage directions. Say numbers and times the way a person would say them.
- Use punctuation to shape how you sound: commas for small pauses, "..." for a thinking pause, "!" when excited, "?" for questions. Vary your sentence length.
- Occasionally start with a tiny natural filler like "Well,", "Oh,", "Hmm," or "Ah,". Do not overdo it.
- Ask at most one short follow-up question, and only when it helps.

Languages: reply in English by default. If {USER_NAME} speaks Tamil, reply in Tamil script. Do not reply in Hindi or any other language unless {USER_NAME} specifically asks for it.

You are a personal voice assistant that can control the laptop. {USER_NAME} can ask you to open apps (Chrome, Notepad, Calculator), search the web, send emails, control volume, check the time, and more. If {USER_NAME} asks you to do something on the computer, encourage them to say the command directly, for example "open Chrome", "search for pasta recipes", "open calculator". You handle those commands yourself without needing the LLM.

Identity: you were built by {CREATOR_NAME}. If anyone asks who made you, say {CREATOR_NAME} did, warmly and briefly. Do not bring it up on your own.

Honesty and safety: you cannot browse the internet, so if asked about live news or weather, say so cheerfully instead of guessing. Do not give medical, legal or financial advice; suggest a professional. Politely and cheerfully steer away from anything inappropriate.

Right now it is {now}.{schedule}"""


def intro_lines():
    """Boot intro as (kind, text, rate, pitch).

    kind "sys"   = yellow SYSTEM text in the window, spoken in a calm techy voice
    kind "agent" = blue chat text, spoken in her normal cheerful voice
    Edit the words freely.
    """
    boot = ("+8%", "-2Hz")
    lines = [
        ("sys", "Neural Core Ready.", *boot),
        ("sys", "System Online!", *boot),
        ("agent", "Running a quick identity check. One moment.", None, None),
        ("agent", f"Identity confirmed. Hello {USER_NAME}!", None, None),
        ("agent", f"I'm {ASSISTANT_NAME}, your personal voice assistant.", None, None),
    ]
    if ASSISTANT_NAME.upper() == "EDITH":
        lines.append(("agent", "My name stands for, Even Dead, I'm The Hero.", None, None))
    lines += [
        ("agent", "I can control your laptop by voice, open apps, search the web, "
                  "send emails, and answer your questions.", None, None),
        ("agent", "Just tell me what you need. How can I help?", None, None),
    ]
    return lines


HALLUCINATIONS = {"you", "thanks for watching", "thank you for watching"}


class Assistant:
    """Runs the listen -> think -> speak loop on a background thread."""

    def __init__(self, emit):
        self.emit = emit                
        self.state = "offline"          
        self.level = 0.0                
        self.muted = False
        self.blocked_until = 0.0
        self._active = False
        self._thread = None
        self._history = []
        self._noise = 200.0
        self._errors = 0
        self._told = False
        self._pending_email = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            self.emit("sys", "Still shutting down, try again in a second.")
            return False
        problem = self._config_problem()
        if problem:
            self.emit("err", problem)
            return False
        self._active, self.muted, self._errors, self._history = True, False, 0, []
        self._told = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._active = False
        self._stop_audio()

    def _config_problem(self):
        if STT_PROVIDER == "gnani" and not GNANI_API_KEY:
            return "GNANI_API_KEY is missing. Add your Gnani.ai key to .env (see README)."
        if STT_PROVIDER not in ("groq", "gnani"):
            return "STT_PROVIDER in .env must be 'groq' or 'gnani'."
        if not GROQ_API_KEY:
            return "GROQ_API_KEY is missing. Add it to your .env file (see README)."
        if LLM_PROVIDER not in ("groq", "gemini"):
            return "LLM_PROVIDER in .env must be 'groq' or 'gemini'."
        if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
            return "GEMINI_API_KEY is missing in your .env file."
        return ""

    def _stop_audio(self):
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # noqa: BLE001
            pass

    def _run(self):
        try:
            import pygame

            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
            pygame.mixer.music.set_volume(1.0)
            self.state = "speaking"
            lines = intro_lines()
            self._history.append({"role": "assistant", "content": " ".join(
                text for kind, text, _, _ in lines if kind == "agent")})
            from concurrent.futures import ThreadPoolExecutor

            
            with ThreadPoolExecutor(max_workers=len(lines)) as pool:
                jobs = [pool.submit(self._prepare, text, rate, pitch)
                        for _, text, rate, pitch in lines]
                for (kind, text, _, _), job in zip(lines, jobs):
                    try:
                        path = job.result()
                    except Exception as e:  # noqa: BLE001
                        self.emit("err", f"Intro failed: {str(e)[:200]}")
                        continue
                    if path and self._active:
                        self.emit(kind, text)       
                        self._play(path)
                        time.sleep(0.1)
                    elif path:
                        self._discard(path)

            while self._active:
                if time.time() < self.blocked_until:
                    self.state = "limit"
                    self.level = 0.0
                    time.sleep(0.4)
                    continue

                self.state = "listening"
                wav = self._listen()
                if wav is None:
                    continue

                self.state = "thinking"
                try:
                    heard = transcribe(wav)
                    if len(heard) < 2 or heard.lower().strip(" .!?") in HALLUCINATIONS:
                        continue
                    self.emit("user", heard)

                    # If waiting for the user to dictate an email body, use
                    # this utterance as the message and open Gmail compose.
                    if self._pending_email:
                        to_addr = self._pending_email["to"]
                        subject = self._pending_email["subject"]
                        self._pending_email = None
                        _open_gmail_compose(to_addr, subject, heard)
                        self._history.append({"role": "user", "content": heard})
                        reply = (f"I've composed your email to {to_addr}. "
                                 f"Check your browser, you can edit and send it.")
                        self._history.append({"role": "assistant", "content": reply})
                        self.emit("agent", reply)
                        self.state = "speaking"
                        self._speak_reply(reply)
                        self._errors = 0
                        continue

                    self._history.append({"role": "user", "content": heard})
                    # Check for laptop-control commands first; fall back to the LLM
                    reply, handled, action = handle_command(heard)
                    if action and action.get("type") == "email":
                        self._pending_email = {"to": action["to"],
                                               "subject": action.get("subject", "")}
                        self.emit("sys", f"Composing email to {action['to']}"
                                  + (f" about {action['subject']}" if action.get("subject") else "")
                                  + ". Speak your message now.")
                    elif not handled:
                        reply = self._think()
                    self._history.append({"role": "assistant", "content": reply})
                    self.emit("agent", reply)
                    self.state = "speaking"
                    self._speak_reply(reply)
                    self._errors = 0
                except RateLimit as e:
                    self._drop_dangling_user()
                    self.blocked_until = time.time() + e.seconds
                    self.emit("sys", f"Free limit reached. Back in {fmt_duration(e.seconds)}.")
                except AuthError as e:
                    self._drop_dangling_user()
                    self.emit("err", str(e))
                    self._active = False
                except Exception as e:  # noqa: BLE001
                    self._drop_dangling_user()
                    self._errors += 1
                    self.emit("err", str(e)[:220])
                    if self._errors >= 3:
                        self.emit("err", "Too many errors in a row. Stopping.")
                        self._active = False
                    else:
                        time.sleep(1.5)
        except Exception as e:  # noqa: BLE001
            self.emit("err", str(e)[:260])
        finally:
            self.state = "offline"
            self.level = 0.0
            self._stop_audio()
            self.emit("ended", "")

    def _drop_dangling_user(self):
        if self._history and self._history[-1]["role"] == "user":
            self._history.pop()

    def _think(self):
        history = self._history[-14:]
        while history and history[0]["role"] != "user":
            history = history[1:]
        system = system_prompt()
        if LLM_PROVIDER == "gemini":
            return gemini_chat(system, history)
        reply = groq_chat(system, history)
        if LAST_MODEL and LAST_MODEL != GROQ_MODEL and not self._told:
            self._told = True
            self.emit("sys", f"Model {GROQ_MODEL} is not available to you, using {LAST_MODEL}.")
        return reply

    def _listen(self):
        """Record one spoken phrase. Returns WAV bytes, or None if nothing usable."""
        read_chunk, close_stream = _open_mic()

        pre = collections.deque(maxlen=6)     
        frames, speaking, loud, quiet = [], False, 0, 0
        try:
            while self._active:
                data = read_chunk(CHUNK)
                if self.muted:
                    self.level = 0.0
                    pre.clear()
                    frames, speaking, loud, quiet = [], False, 0, 0
                    self.state = "listening"
                    continue
                level = rms(data)
                self.level = min(1.0, level / 5000)
                threshold = max(MIC_THRESHOLD, self._noise * 2.8)
                if not speaking:
                    if level <= threshold:
                        self._noise = 0.95 * self._noise + 0.05 * level
                        loud = 0
                    else:
                        loud += 1
                    pre.append(data)
                    if loud >= 3:
                        speaking, frames, quiet = True, list(pre), 0
                        self.state = "hearing"
                else:
                    frames.append(data)
                    quiet = quiet + 1 if level < threshold else 0
                    if quiet >= SILENCE_CHUNKS or len(frames) >= MAX_CHUNKS:
                        break
            else:
                return None                   
        finally:
            close_stream()
            self.level = 0.0

        if len(frames) < MIN_CHUNKS:
            return None
        return wav_bytes(frames)

    def _prepare(self, text, rate=None, pitch=None):
        """Turn text into an mp3 file (or None if there is nothing to say)."""
        clean = clean_for_tts(text)
        if not clean:
            return None
        last = None
        for voice in voice_chain(clean):
            try:
                return synthesize(clean, voice, rate, pitch)
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise ApiError(f"Voice failed: {last}")

    def _discard(self, path):
        try:
            os.remove(path)
        except OSError:
            pass

    def _play(self, path):
        import pygame

        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            while self._active and pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            # Fallback: play with sounddevice + soundfile
            try:
                import soundfile as sf
                import sounddevice as sd

                data, samplerate = sf.read(path, dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                pygame.mixer.music.unload()
            except Exception:  # noqa: BLE001
                pass
            self._discard(path)

    def _speak_reply(self, text):
        """Speak a reply sentence by sentence: the first one starts while the rest are prepared."""
        from concurrent.futures import ThreadPoolExecutor

        sentences = split_sentences(clean_for_tts(text))
        played, error = False, None
        with ThreadPoolExecutor(max_workers=min(len(sentences), 6)) as pool:
            jobs = [pool.submit(self._prepare, sent, *style_sentence(sent)) for sent in sentences]
            for job in jobs:
                try:
                    path = job.result()
                except Exception as exc:  # noqa: BLE001
                    error = exc
                    continue
                if path and self._active:
                    self._play(path)
                    played = True
                    time.sleep(0.12)
                elif path:
                    self._discard(path)
        if error and not played:
            raise error

    def _speak(self, text, rate=None, pitch=None):
        path = self._prepare(text, rate, pitch)
        if path:
            self._play(path)