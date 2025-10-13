import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

from .multimodality_panel import MultimodalityPanel
from .vlm_pipeline_operator import RunVLMPipeline

def register(plugin):
    """Register operators with the plugin."""
    plugin.register(MultimodalityPanel)
    plugin.register(RunVLMPipeline)