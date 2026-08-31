import argparse
import json
import os
import random
import sys
import traceback

PROTOCOL_PREFIX = "@@COSYVOICE2@@"


def _prepare_runtime(runtime_dir: str):
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_dir = os.path.abspath(runtime_dir)
    matcha_dir = os.path.join(runtime_dir, "third_party", "Matcha-TTS")

    cleaned = []
    for entry in sys.path:
        resolved = os.path.abspath(entry or os.getcwd())
        if resolved != plugin_dir:
            cleaned.append(entry)
    sys.path[:] = cleaned
    sys.path.insert(0, runtime_dir)
    if os.path.isdir(matcha_dir):
        sys.path.insert(0, matcha_dir)


def _set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_audio(path: str, waveform, sample_rate: int):
    import soundfile as sf

    audio = waveform.detach().float().cpu().squeeze(0).numpy()
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    _prepare_runtime(args.runtime_dir)

    from cosyvoice.cli.cosyvoice import CosyVoice2

    model = None

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
            if request.get("action") == "shutdown":
                print(PROTOCOL_PREFIX + json.dumps({"ok": True}), flush=True)
                return

            if request.get("action") != "synthesize":
                raise ValueError("unsupported worker action")

            if model is None:
                model = CosyVoice2(
                    args.model_dir,
                    load_jit=False,
                    load_trt=False,
                    load_vllm=False,
                    fp16=False,
                )

            tts_text = str(request.get("tts_text") or "").strip()
            prompt_text = str(request.get("prompt_text") or "").strip()
            instruct_text = str(request.get("instruct_text") or "").strip()
            prompt_wav = request["prompt_wav"]
            output_wav = request["output_wav"]
            speed = float(request.get("speed", 1.0))
            seed = int(request.get("seed", 42))

            if not tts_text:
                raise ValueError("tts_text 不能为空")

            _set_seed(seed)

            if instruct_text:
                output = model.inference_instruct2(
                    tts_text,
                    instruct_text,
                    prompt_wav,
                    stream=False,
                    speed=speed,
                )
                mode = "instruct2"
            elif prompt_text:
                output = model.inference_zero_shot(
                    tts_text,
                    prompt_text,
                    prompt_wav,
                    stream=False,
                    speed=speed,
                )
                mode = "zero_shot"
            else:
                output = model.inference_cross_lingual(
                    tts_text,
                    prompt_wav,
                    stream=False,
                    speed=speed,
                )
                mode = "cross_lingual"

            chunks = [item["tts_speech"] for item in output]
            if not chunks:
                raise RuntimeError("CosyVoice2 没有生成任何音频")

            import torch

            waveform = torch.cat(chunks, dim=1)
            _write_audio(output_wav, waveform, int(model.sample_rate))
            response = {
                "ok": True,
                "sample_rate": int(model.sample_rate),
                "mode": mode,
            }
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        print(PROTOCOL_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
