import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict

import folder_paths
import librosa
import soundfile as sf
import torch
import torchaudio
from aiohttp import web
from server import PromptServer

VOICE_TYPE = "COSYVOICE_VOICE"
VOICE_FORMAT = "cosyvoice2_reference_voice"
VOICE_FORMAT_VERSION = 2
VOICE_MODE_ZERO_SHOT = "zero_shot"
VOICE_MODE_CROSS_LINGUAL = "cross_lingual"
VOICE_UPLOAD_SUBDIR = os.path.join("CosyVoice", "voices")
MAX_PROMPT_SECONDS = 15.0

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODELS = os.path.join(PLUGIN_DIR, "pretrained_models")
COSYVOICE2_RUNTIME_DIR = os.path.join(PRETRAINED_MODELS, "CosyVoice2-runtime")
COSYVOICE2_MODEL_DIR = os.path.join(PRETRAINED_MODELS, "CosyVoice2-0.5B")
COSYVOICE2_MODEL_ID = "iic/CosyVoice2-0.5B"
COSYVOICE2_REPOSITORY = "https://github.com/QwenAudio/CosyVoice.git"
COSYVOICE2_RUNTIME_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
WORKER_PROTOCOL_PREFIX = "@@COSYVOICE2@@"

_RUNTIME_LOCK = threading.Lock()


def _input_dir() -> str:
    path = folder_paths.get_input_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _output_dir() -> str:
    path = folder_paths.get_output_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _voice_mode(voice: Dict[str, Any]) -> str:
    return voice.get("mode", VOICE_MODE_ZERO_SHOT)


def _validate_voice(voice: Dict[str, Any]) -> None:
    if not isinstance(voice, dict):
        raise ValueError("voice 必须是 CosyVoice 音色对象")

    if voice.get("format") != VOICE_FORMAT or int(voice.get("format_version", 0)) != VOICE_FORMAT_VERSION:
        raise ValueError(
            "这是旧版 CosyVoice 音色文件，CosyVoice2/instruct2 需要保留原始参考音频；"
            "请使用 CosyVoice - Extract Voice 重新抽取并保存 .pt"
        )

    mode = _voice_mode(voice)
    if mode not in {VOICE_MODE_ZERO_SHOT, VOICE_MODE_CROSS_LINGUAL}:
        raise ValueError(f"不支持的音色模式: {mode!r}")

    prompt_wav = voice.get("prompt_wav")
    if not isinstance(prompt_wav, torch.Tensor) or prompt_wav.ndim != 2 or prompt_wav.shape[0] != 1:
        raise ValueError("音色缺少有效的 16kHz 单声道 prompt_wav")
    if int(voice.get("sample_rate", 0)) != 16000:
        raise ValueError("音色 prompt_wav 必须为 16000Hz")
    if not isinstance(voice.get("prompt_text", ""), str):
        raise ValueError("音色 prompt_text 格式无效")


def _cpu_voice(voice: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(voice)
    for key, value in list(result.items()):
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().contiguous()
    return result


def _load_voice_file(path: str) -> Dict[str, Any]:
    try:
        voice = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        voice = torch.load(path, map_location="cpu")
    _validate_voice(voice)
    return _cpu_voice(voice)


def _list_pt_files(base_dir: str):
    files = []
    for root, _, names in os.walk(base_dir):
        for name in names:
            if name.lower().endswith(".pt"):
                full_path = os.path.join(root, name)
                files.append(os.path.relpath(full_path, base_dir).replace(os.sep, "/"))
    return sorted(files)


def _list_voice_files():
    files = [f"input/{path}" for path in _list_pt_files(_input_dir())]
    files.extend(f"output/{path}" for path in _list_pt_files(_output_dir()))
    return sorted(files)


def _resolve_voice_path(voice_file: str) -> str:
    value = (voice_file or "").replace("\\", "/")
    if value.startswith("input/"):
        base_dir = os.path.abspath(_input_dir())
        relative = value[len("input/"):]
    elif value.startswith("output/"):
        base_dir = os.path.abspath(_output_dir())
        relative = value[len("output/"):]
    else:
        raise ValueError("voice_file 必须来自 input/ 或 output/ 目录")

    path = os.path.abspath(os.path.join(base_dir, relative))
    if os.path.commonpath([base_dir, path]) != base_dir:
        raise ValueError("voice_file 超出允许目录范围")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"音色文件不存在: {voice_file}")
    return path


def _safe_upload_filename(filename: str) -> str:
    name = os.path.basename(filename or "voice.pt")
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".pt":
        raise ValueError("只允许上传 .pt 音色文件")
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem).strip("._") or "voice"
    return safe_stem[:128] + ".pt"


def _unique_path(directory: str, filename: str):
    stem, ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    index = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{index:03d}{ext}")
        index += 1
    return path


def _audio_to_16k(audio):
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    source_sr = int(audio["sample_rate"])
    speech = waveform.float().mean(dim=0, keepdim=True).cpu()
    if source_sr != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=16000)(speech)

    trimmed, _ = librosa.effects.trim(speech.squeeze(0).numpy(), top_db=60)
    if trimmed.size:
        speech = torch.from_numpy(trimmed).float().unsqueeze(0)

    max_samples = int(16000 * MAX_PROMPT_SECONDS)
    if speech.shape[1] > max_samples:
        speech = speech[:, :max_samples]

    peak = float(speech.abs().max()) if speech.numel() else 0.0
    if peak <= 0:
        raise ValueError("参考音频为空或没有有效声音")
    if peak > 0.8:
        speech = speech / peak * 0.8

    return speech.contiguous()


def _make_voice(prompt_wav: torch.Tensor, prompt_text: str) -> Dict[str, Any]:
    prompt_text = (prompt_text or "").strip()
    voice = {
        "format": VOICE_FORMAT,
        "format_version": VOICE_FORMAT_VERSION,
        "model": "CosyVoice2-0.5B",
        "mode": VOICE_MODE_ZERO_SHOT if prompt_text else VOICE_MODE_CROSS_LINGUAL,
        "sample_rate": 16000,
        "prompt_wav": prompt_wav.detach().cpu().contiguous(),
        "prompt_text": prompt_text,
    }
    _validate_voice(voice)
    return voice


def _run_git(args):
    try:
        process = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 git，请先安装 Git 并确保 git 在 PATH 中") from exc

    if process.returncode != 0:
        raise RuntimeError(f"Git 命令失败: git {' '.join(args)}\n{process.stdout}")
    return process.stdout


def _ensure_cosyvoice2_runtime():
    with _RUNTIME_LOCK:
        os.makedirs(PRETRAINED_MODELS, exist_ok=True)
        runtime_entry = os.path.join(COSYVOICE2_RUNTIME_DIR, "cosyvoice", "cli", "cosyvoice.py")

        if not os.path.isfile(runtime_entry):
            if os.path.exists(COSYVOICE2_RUNTIME_DIR):
                raise RuntimeError(
                    f"CosyVoice2 runtime 目录不完整，请删除后重试: {COSYVOICE2_RUNTIME_DIR}"
                )
            _run_git(["clone", COSYVOICE2_REPOSITORY, COSYVOICE2_RUNTIME_DIR])

        current_commit = _run_git(["-C", COSYVOICE2_RUNTIME_DIR, "rev-parse", "HEAD"]).strip()
        if current_commit != COSYVOICE2_RUNTIME_COMMIT:
            _run_git(["-C", COSYVOICE2_RUNTIME_DIR, "fetch", "origin", COSYVOICE2_RUNTIME_COMMIT])
            _run_git(["-C", COSYVOICE2_RUNTIME_DIR, "checkout", "--detach", COSYVOICE2_RUNTIME_COMMIT])

        _run_git(["-C", COSYVOICE2_RUNTIME_DIR, "submodule", "update", "--init", "--recursive"])

        model_config = os.path.join(COSYVOICE2_MODEL_DIR, "cosyvoice2.yaml")
        if not os.path.isfile(model_config):
            from modelscope import snapshot_download

            snapshot_download(model_id=COSYVOICE2_MODEL_ID, local_dir=COSYVOICE2_MODEL_DIR)


class _CosyVoice2Worker:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()

    def _start(self):
        _ensure_cosyvoice2_runtime()

        if self.process is not None and self.process.poll() is None:
            return

        worker_script = os.path.join(PLUGIN_DIR, "cosyvoice2_worker.py")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"

        self.process = subprocess.Popen(
            [
                sys.executable,
                worker_script,
                "--runtime-dir",
                COSYVOICE2_RUNTIME_DIR,
                "--model-dir",
                COSYVOICE2_MODEL_DIR,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            cwd=COSYVOICE2_RUNTIME_DIR,
        )

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._start()
            if self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("CosyVoice2 worker 管道初始化失败")

            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

            logs = []
            while True:
                line = self.process.stdout.readline()
                if line == "":
                    code = self.process.poll()
                    raise RuntimeError(
                        f"CosyVoice2 worker 异常退出，code={code}\n" + "".join(logs[-50:])
                    )

                if line.startswith(WORKER_PROTOCOL_PREFIX):
                    response = json.loads(line[len(WORKER_PROTOCOL_PREFIX):])
                    if not response.get("ok"):
                        raise RuntimeError(
                            "CosyVoice2 推理失败: "
                            + response.get("error", "unknown error")
                            + "\n"
                            + response.get("traceback", "")
                        )
                    return response

                logs.append(line)
                print(line, end="")

    def close(self):
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                process.stdin.flush()
            process.terminate()
        except Exception:
            pass


_GLOBAL_WORKER = _CosyVoice2Worker()
atexit.register(_GLOBAL_WORKER.close)


@PromptServer.instance.routes.get("/cosyvoice/voice_files")
async def cosyvoice_voice_files(_request):
    return web.json_response({"files": _list_voice_files()})


@PromptServer.instance.routes.post("/cosyvoice/upload_voice")
async def cosyvoice_upload_voice(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file" or not field.filename:
        return web.json_response({"error": "缺少上传文件"}, status=400)

    try:
        filename = _safe_upload_filename(field.filename)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    upload_dir = os.path.join(_input_dir(), VOICE_UPLOAD_SUBDIR)
    os.makedirs(upload_dir, exist_ok=True)
    path = _unique_path(upload_dir, filename)

    try:
        with open(path, "wb") as output:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        _load_voice_file(path)
    except Exception as exc:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return web.json_response({"error": f"无效的 CosyVoice2 音色文件: {exc}"}, status=400)

    relative = os.path.relpath(path, _input_dir()).replace(os.sep, "/")
    return web.json_response({"name": f"input/{relative}"})


class CosyVoiceExtractVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt_wav": ("AUDIO",)},
            "optional": {"prompt_text": ("STRING", {"multiline": True, "default": ""})},
        }

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "extract"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = (
        "创建 CosyVoice2 持久化音色。参考音频会转为 16kHz 单声道并保存在 .pt 中；"
        "prompt_text 有值时用于 zero-shot，为空时使用 cross-lingual。"
    )

    @torch.no_grad()
    def extract(self, prompt_wav, prompt_text=""):
        return (_make_voice(_audio_to_16k(prompt_wav), prompt_text),)


class CosyVoiceSaveVoiceNode:
    def __init__(self):
        self.output_dir = _output_dir()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "voice": (VOICE_TYPE,),
            "filename_prefix": ("STRING", {"default": "CosyVoice/voice"}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = "将 CosyVoice2 持久化音色保存到 output，支持子目录和自动计数。"

    def save(self, voice, filename_prefix):
        _validate_voice(voice)
        full_output_folder, filename, counter, _subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, 0, 0
        )
        os.makedirs(full_output_folder, exist_ok=True)
        filename = filename.replace("%batch_num%", "0")
        path = os.path.join(full_output_folder, f"{filename}_{counter:05d}_.pt")
        torch.save(_cpu_voice(voice), path)
        relative_path = os.path.relpath(path, self.output_dir).replace(os.sep, "/")
        return {"ui": {"text": [relative_path]}}


class CosyVoiceLoadVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_voice_files() or ["(暂无音色，可点击上传 .pt)"]
        return {"required": {"voice_file": (files,)}}

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "load"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = "选择 input/output 中的 CosyVoice2 .pt 音色，或使用节点按钮上传。"

    @classmethod
    def IS_CHANGED(cls, voice_file):
        if not voice_file or voice_file.startswith("("):
            return float("nan")
        try:
            path = _resolve_voice_path(voice_file)
            stat = os.stat(path)
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except (OSError, ValueError, FileNotFoundError):
            return float("nan")

    def load(self, voice_file):
        if not voice_file or voice_file.startswith("("):
            raise FileNotFoundError("没有可加载的音色文件，请点击上传 .pt 或先使用 Save Voice 保存音色")
        return (_load_voice_file(_resolve_voice_path(voice_file)),)


class CosyVoiceVoiceTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_text": ("STRING", {"multiline": True, "default": "你好，这是使用已保存音色生成的语音。"}),
                "voice": (VOICE_TYPE,),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "instruct_text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = (
        "使用官方 CosyVoice2-0.5B runtime。instruct_text 有值时调用 inference_instruct2；"
        "否则根据音色是否包含 prompt_text 自动调用 zero-shot/cross-lingual。"
    )

    @torch.no_grad()
    def generate(self, tts_text, voice, speed, seed, instruct_text=""):
        _validate_voice(voice)
        tts_text = (tts_text or "").strip()
        instruct_text = (instruct_text or "").strip()
        if not tts_text:
            raise ValueError("tts_text 不能为空")

        with tempfile.TemporaryDirectory(prefix="cosyvoice2_") as temp_dir:
            prompt_path = os.path.join(temp_dir, "prompt.wav")
            output_path = os.path.join(temp_dir, "output.wav")

            prompt_audio = voice["prompt_wav"].detach().float().cpu().squeeze(0).numpy()
            sf.write(prompt_path, prompt_audio, 16000, subtype="PCM_16")

            response = _GLOBAL_WORKER.request({
                "action": "synthesize",
                "tts_text": tts_text,
                "prompt_text": voice.get("prompt_text", ""),
                "instruct_text": instruct_text,
                "prompt_wav": prompt_path,
                "output_wav": output_path,
                "speed": float(speed),
                "seed": int(seed),
            })

            waveform, sample_rate = sf.read(output_path, dtype="float32", always_2d=True)
            audio_tensor = torch.from_numpy(waveform.T.copy()).unsqueeze(0)
            return ({
                "waveform": audio_tensor,
                "sample_rate": int(response.get("sample_rate", sample_rate)),
            },)
