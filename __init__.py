import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

from .multimodality_panel import MultimodalityPanel

def register(plugin):
    """Register operators with the plugin."""
    plugin.register(MultimodalityPanel) 