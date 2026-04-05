"""stream() vs generate() 성능 비교.

사용법:
    cd hayakoe/
    python dev-tools/test_stream.py
"""

import time
import wave

import numpy as np

from hayakoe import TTS


TEXT = (
    "日本の四季は、世界中の人々を魅了する美しさを持っています。"
    "春になると、全国各地で桜が咲き誇り、人々は花見を楽しみながら、新しい季節の訪れを祝います。"
    "公園や川沿いには、家族連れや友人同士が集まり、桜の木の下でお弁当を広げ、笑い声が絶えません。"
    "夏には、蒸し暑い日々が続きますが、花火大会や夏祭りが各地で開催され、浴衣姿の人々が夜の街を彩ります。"
    "風鈴の涼やかな音色が、暑さの中にひとときの清涼感をもたらしてくれます。"
    "秋になると、山々は赤や黄色に染まり、紅葉狩りを楽しむ人々で賑わいます。"
    "澄んだ空気の中で食べる秋の味覚、例えば栗や秋刀魚、松茸などは格別です。"
    "そして冬が訪れると、北国では雪が降り積もり、温泉地には多くの観光客が訪れます。"
    "露天風呂から眺める雪景色は、まさに日本ならではの贅沢な体験と言えるでしょう。"
    "このように、日本では一年を通じて季節ごとの風情を楽しむ文化が深く根付いており、"
    "それぞれの季節に合わせた行事や食文化が、日常生活を豊かに彩っています。"
)


def save_wav(path: str, data: np.ndarray, sr: int = 44100) -> None:
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def main():
    device = "cuda"
    speaker = TTS(device=device).load("jvnv-F1-jp")

    # 워밍업
    print("워밍업...")
    speaker.generate("テスト。").save("/dev/null")

    # ── stream() ──
    print(f"\n{'='*60}")
    print("  stream()")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    chunks = []
    for i, chunk in enumerate(speaker.stream(TEXT)):
        elapsed = time.perf_counter() - t0
        chunk_sec = len(chunk.data) / chunk.sr
        print(f"  chunk {i}: {chunk_sec:.2f}s audio (@ {elapsed:.2f}s)")
        chunks.append(chunk.data)

    all_audio = np.concatenate(chunks)
    stream_time = time.perf_counter() - t0
    sr = speaker._hps.data.sampling_rate
    audio_sec = len(all_audio) / sr
    print(f"\n  총 합성: {stream_time:.3f}s, 오디오: {audio_sec:.1f}s, RTF: {stream_time/audio_sec:.3f}")
    save_wav("output_stream.wav", all_audio, sr)

    # ── generate() ──
    print(f"\n{'='*60}")
    print("  generate()")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = speaker.generate(TEXT)
    gen_time = time.perf_counter() - t0
    audio_sec = len(result.data) / result.sr
    print(f"  총 합성: {gen_time:.3f}s, 오디오: {audio_sec:.1f}s, RTF: {gen_time/audio_sec:.3f}")
    result.save("output_generate.wav")

    # ── 비교 ──
    print(f"\n{'='*60}")
    print("  비교")
    print(f"{'='*60}")
    print(f"  stream():   {stream_time:.3f}s")
    print(f"  generate(): {gen_time:.3f}s")
    if gen_time > 0:
        print(f"  속도 향상:  {gen_time / stream_time:.1f}x")


if __name__ == "__main__":
    main()
