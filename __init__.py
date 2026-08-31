from .nodes import TextNode, CosyVoiceNode, LoadSRT, CosyVoiceDubbingNode
from .persistent_voice_nodes import (
    CosyVoiceExtractZeroShotVoiceNode,
    CosyVoiceSaveVoiceNode,
    CosyVoiceLoadVoiceNode,
    CosyVoiceZeroShotVoiceTTSNode,
    CosyVoiceCrossLingualVoiceTTSNode,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT": LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode": CosyVoiceDubbingNode,
    "CosyVoiceExtractZeroShotVoiceNode": CosyVoiceExtractZeroShotVoiceNode,
    "CosyVoiceSaveVoiceNode": CosyVoiceSaveVoiceNode,
    "CosyVoiceLoadVoiceNode": CosyVoiceLoadVoiceNode,
    "CosyVoiceZeroShotVoiceTTSNode": CosyVoiceZeroShotVoiceTTSNode,
    "CosyVoiceCrossLingualVoiceTTSNode": CosyVoiceCrossLingualVoiceTTSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceExtractZeroShotVoiceNode": "CosyVoice - Extract Zero-shot Voice",
    "CosyVoiceSaveVoiceNode": "CosyVoice - Save Voice",
    "CosyVoiceLoadVoiceNode": "CosyVoice - Load Voice",
    "CosyVoiceZeroShotVoiceTTSNode": "CosyVoice - Zero-shot Voice TTS",
    "CosyVoiceCrossLingualVoiceTTSNode": "CosyVoice - Cross-lingual Voice TTS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
