#!/usr/bin/env python3

import sys
import time
import soundfile as sf
from pathlib import Path
from kokoro_onnx import Kokoro

def main(kokoro, file_path, lang, voice):
    filename = Path(file_path).name
    texts = [read_file(file_path)]

    total_chars = sum([len(t) for t in texts])
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {total_chars:,}')
    print('Total words:', len(' '.join(texts).split(' ')))

    i = 1
    wav_files = []
    for text in texts:
        wav_filename = filename + ".wav"
        wav_files.append(wav_filename)
        if Path(wav_filename).exists():
            print(f'File for chapter {i} already exists. Skipping')
            i += 1
            continue
        print(f'Reading file {i} ({len(text):,} characters)...')

        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
        sf.write(f'{wav_filename}', samples, sample_rate)

def read_file(file):
    handler = open(file, "r")
    content = handler.read()
    handler.close()

    return content

def cli_main():
    if not Path('kokoro-v1.0.onnx').exists() or not Path('voices-v1.0.bin').exists():
        print('Error: kokoro-v1.0.onnx and voices-v1.0.bin must be in the current directory. Please download them.')
        sys.exit(1)
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    voices = list(kokoro.get_voices())
    default_voice = 'af_sky' if 'af_sky' in voices else voices[0]
    filename = sys.argv[1]

    file = Path(filename)
    if not file.is_file():
        raise FileNotFoundError(f'File {filename} not found')

    main(kokoro, filename, "en-gb", default_voice)

if __name__ == '__main__':
    cli_main()

