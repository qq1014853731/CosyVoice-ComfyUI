import os
from typing import Any, Dict

import folder_paths
import torch
import torchaudio
from aiohttp import web
from server import PromptServer

from .nodes import CosyVoice, postprocess, pretrained_models, set_all_random_seed, speed_change, target_sr

VOICE_TYPE = "COSYVOICE_VOICE"
VOICE_FORMAT = "cosyvoice_voice_condition"
VOICE_FORMAT_LEGACY = "cosyvoice_zero_shot_voice"
VOICE_FORMAT_VERSION = 1
VOICE_MODE_ZERO_SHOT = "zero_shot"
VOICE_MODE_CROSS_LINGUAL = "cross_lingual"
VOICE_UPLOAD_SUBDIR = os.path.join("CosyVoice", "voices")


class _CosyVoiceLoader:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None

    def get(self):
        model_dir = os.path.join(pretrained_models, "CosyVoice-300M")
        if self.cosyvoice is None or self.model_dir != model_dir:
            from modelscope import snapshot_download
            snapshot_download(model_id="iic/CosyVoice-300M", local_dir=model_dir)
            self.cosyvoice = CosyVoice(model_dir)
            self.model_dir = model_dir
        return self.cosyvoice


_GLOBAL_LOADER = _CosyVoiceLoader()


def _input_dir() -> str:
    path = folder_paths.get_input_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _output_dir() -> str:
    path = folder_paths.get_output_directory()
    os.makedirs(path, exist_ok=True)
    return path


def _voice_mode(voice: Dict[str, Any]) -> str:
    # 兼容本功能第一版生成的 zero-shot .pt：当时没有 mode 字段。
    return voice.get("mode", VOICE_MODE_ZERO_SHOT)


def _validate_voice(voice: Dict[str, Any], expected_mode: str | None = None) -> None:
    if not isinstance(voice, dict):
        raise ValueError("voice 必须是 CosyVoice 音色对象")

    voice_format = voice.get("format")
    if voice_format not in {VOICE_FORMAT, VOICE_FORMAT_LEGACY}:
        raise ValueError(f"不支持的音色格式: {voice_format!r}")
    if int(voice.get("format_version", 0)) != VOICE_FORMAT_VERSION:
        raise ValueError(f"不支持的音色格式版本: {voice.get('format_version')!r}")

    mode = _voice_mode(voice)
    if mode not in {VOICE_MODE_ZERO_SHOT, VOICE_MODE_CROSS_LINGUAL}:
        raise ValueError(f"不支持的音色模式: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"音色模式不匹配: 当前为 {mode}，该节点需要 {expected_mode}")

    common = (
        "flow_prompt_speech_token", "flow_prompt_speech_token_len",
        "prompt_speech_feat", "prompt_speech_feat_len",
        "llm_embedding", "flow_embedding",
    )
    required = list(common)
    if mode == VOICE_MODE_ZERO_SHOT:
        required.extend((
            "prompt_text", "prompt_text_len",
            "llm_prompt_speech_token", "llm_prompt_speech_token_len",
        ))

    missing = [key for key in required if not isinstance(voice.get(key), torch.Tensor)]
    if missing:
        raise ValueError(f"音色缺少 Tensor: {', '.join(missing)}")


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
    waveform = audio["waveform"].squeeze(0)
    source_sr = int(audio["sample_rate"])
    speech = waveform.mean(dim=0, keepdim=True)
    if source_sr != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=16000)(speech)
    return postprocess(speech)


def _make_voice(model_input: Dict[str, Any], mode: str) -> Dict[str, Any]:
    model_input = dict(model_input)
    model_input.pop("text", None)
    model_input.pop("text_len", None)
    voice = {
        "format": VOICE_FORMAT,
        "format_version": VOICE_FORMAT_VERSION,
        "model": "CosyVoice-300M",
        "mode": mode,
        **_cpu_voice(model_input),
    }
    _validate_voice(voice, expected_mode=mode)
    return voice


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
        return web.json_response({"error": f"无效的 CosyVoice 音色文件: {exc}"}, status=400)

    relative = os.path.relpath(path, _input_dir()).replace(os.sep, "/")
    return web.json_response({"name": f"input/{relative}"})


class CosyVoiceExtractZeroShotVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            "prompt_wav": ("AUDIO",),
        }}

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "extract"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = "提取 CosyVoice zero-shot 音色。prompt_text 必须与参考音频内容匹配。"

    @torch.no_grad()
    def extract(self, prompt_text, prompt_wav):
        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text 不能为空，zero-shot 音色需要参考音频对应文本")

        cosyvoice = _GLOBAL_LOADER.get()
        prompt_text = cosyvoice.frontend.text_normalize(prompt_text, split=False)
        prompt_speech_16k = _audio_to_16k(prompt_wav)
        model_input = cosyvoice.frontend.frontend_zero_shot("", prompt_text, prompt_speech_16k)
        return (_make_voice(model_input, VOICE_MODE_ZERO_SHOT),)


class CosyVoiceExtractCrossLingualVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_wav": ("AUDIO",)}}

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "extract"
    CATEGORY = "AIFSH_CosyVoice/voice"
    DESCRIPTION = "提取 CosyVoice cross-lingual 音色，只需要参考音频，不需要 prompt_text。"

    @torch.no_grad()
    def extract(self, prompt_wav):
        cosyvoice = _GLOBAL_LOADER.get()
        prompt_speech_16k = _audio_to_16k(prompt_wav)

        # 与官方 frontend_cross_lingual() 保持一致：先构造 zero-shot conditioning，
        # 再移除 LLM 侧的 prompt text / prompt speech token，仅保留跨语种推理需要的条件。
        model_input = cosyvoice.frontend.frontend_zero_shot("", "", prompt_speech_16k)
        model_input.pop("prompt_text", None)
        model_input.pop("prompt_text_len", None)
        model_input.pop("llm_prompt_speech_token", None)
        model_input.pop("llm_prompt_speech_token_len", None)
        return (_make_voice(model_input, VOICE_MODE_CROSS_LINGUAL),)


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
    DESCRIPTION = "按 SaveImage 风格将 CosyVoice 音色保存到 output，支持子目录和自动计数。"

    def save(self, voice, filename_prefix):
        _validate_voice(voice)
        full_output_folder, filename, counter, _subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, 0, 0
        )
        os.makedirs(full_output_folder, exist_ok=True)
        filename = filename.replace("%batch_num%", "0")
        path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.pt")
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
    DESCRIPTION = "选择 input/output 中的 .pt 音色，或使用节点上的上传按钮从本地上传。"

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


class _CosyVoicePersistentTTSBase:
    mode = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tts_text": ("STRING", {"multiline": True, "default": "你好，这是使用已保存音色生成的语音。"}),
            "voice": (VOICE_TYPE,),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            "seed": ("INT", {"default": 42}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIFSH_CosyVoice/voice"

    @torch.no_grad()
    def generate(self, tts_text, voice, speed, seed):
        _validate_voice(voice, expected_mode=self.mode)
        cosyvoice = _GLOBAL_LOADER.get()
        device = cosyvoice.model.device
        set_all_random_seed(seed)

        conditioning = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in voice.items()
            if key not in {"format", "format_version", "model", "mode"}
        }

        output_list = []
        for text in cosyvoice.frontend.text_normalize(tts_text, split=True):
            text_token, text_token_len = cosyvoice.frontend._extract_text_token(text)
            model_input = {"text": text_token, "text_len": text_token_len, **conditioning}
            for out_dict in cosyvoice.model.inference(**model_input, stream=False):
                output_numpy = out_dict["tts_speech"].squeeze(0).numpy() * 32768
                output_numpy = output_numpy.astype("int16")
                if speed != 1.0:
                    output_numpy = speed_change(output_numpy, speed, target_sr)
                output_list.append(torch.tensor(output_numpy / 32768.0, dtype=torch.float32).unsqueeze(0))

        if not output_list:
            raise ValueError("没有生成任何音频")

        audio = {"waveform": torch.cat(output_list, dim=1).unsqueeze(0), "sample_rate": target_sr}
        return (audio,)


class CosyVoiceZeroShotVoiceTTSNode(_CosyVoicePersistentTTSBase):
    mode = VOICE_MODE_ZERO_SHOT
    DESCRIPTION = "使用持久化 zero-shot 音色生成语音；音色创建时需要 prompt_text + prompt_wav。"


class CosyVoiceCrossLingualVoiceTTSNode(_CosyVoicePersistentTTSBase):
    mode = VOICE_MODE_CROSS_LINGUAL
    DESCRIPTION = "使用持久化 cross-lingual 音色生成语音；音色创建时只需要 prompt_wav。"
