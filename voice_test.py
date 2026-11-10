"""Interactive voice tester: pick a voice, type any sentence, hear it.

Run:  python voice_test.py
Tip: you can paste Hindi or Assamese text too, to check pronunciation.
"""
import os
import time

import pygame

from assistant_core import ASSISTANT_NAME, USER_NAME, synthesize

VOICES = [
    ("British, Sonia (polished, Jarvis-style)", "en-GB-SoniaNeural"),
    ("British, Libby (softer, friendly)", "en-GB-LibbyNeural"),
    ("Indian English, Neerja", "en-IN-NeerjaNeural"),
    ("American, Ava (bright)", "en-US-AvaMultilingualNeural"),
    ("American, Emma (warm)", "en-US-EmmaMultilingualNeural"),
    ("American, Jenny (casual)", "en-US-JennyNeural"),
    ("American, Aria (expressive)", "en-US-AriaNeural"),
    ("American, Michelle (calm)", "en-US-MichelleNeural"),
    ("Australian, Natasha", "en-AU-NatashaNeural"),
    ("Irish, Emily", "en-IE-EmilyNeural"),
    ("South African, Leah", "en-ZA-LeahNeural"),
    ("Tamil, Pallavi", "ta-IN-PallaviNeural"),
    ("Tamil, Valluvar", "ta-IN-ValluvarNeural"),
]

DEFAULT_TEXT = f"Hi {USER_NAME}! I'm {ASSISTANT_NAME}. It's so nice to meet you. How can I help you today?"

pygame.mixer.init()


def play(text, voice):
    try:
        path = synthesize(text, voice)
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not load this voice: {exc}")
        return
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    os.remove(path)


while True:
    print("\n=== VOICES ===")
    for i, (label, _) in enumerate(VOICES, 1):
        print(f"{i:>2}  {label}")
    choice = input("\nType a voice number (or q to quit): ").strip().lower()
    if choice in ("q", "quit"):
        break
    if not choice.isdigit() or not 1 <= int(choice) <= len(VOICES):
        print("Please type a number from the list.")
        continue
    label, voice = VOICES[int(choice) - 1]
    print(f"\nVoice: {label}")
    print("Press Enter to hear the sample sentence, or type/paste your own text.")
    print("Type b to go back to the list.")
    while True:
        text = input("> ").strip()
        if text.lower() == "b":
            break
        play(text or DEFAULT_TEXT, voice)

print("Bye! Tell me which voice you liked best.")