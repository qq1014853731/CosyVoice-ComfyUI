from .nodes import TextNode, CosyVoiceNode, LoadSRT, CosyVoiceDubbingNode
from .persistent_voice_nodes import (
    CosyVoiceExtractVoiceNode,
    CosyVoiceSaveVoiceNode,
    CosyVoiceLoadVoiceNode,
    CosyVoiceVoiceTTSNode,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT": LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode": CosyVoiceDubbingNode,
    "CosyVoiceExtractVoiceNode": CosyVoiceExtractVoiceNode,
    "CosyVoiceSaveVoiceNode": CosyVoiceSaveVoiceNode,
    "CosyVoiceLoadVoiceNode": CosyVoiceLoadVoiceNode,
    "CosyVoiceVoiceTTSNode": CosyVoiceVoiceTTSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceExtractVoiceNode": "CosyVoice - Extract Voice",
    "CosyVoiceSaveVoiceNode": "CosyVoice - Save Voice",
    "CosyVoiceLoadVoiceNode": "CosyVoice - Load Voice",
    "CosyVoiceVoiceTTSNode": "CosyVoice - Voice TTS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
