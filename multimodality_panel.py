import fiftyone.operators as foo
from fiftyone.operators.types import View, Object, Choices, Property, GridView, TableView, Object as TypeObject
import fiftyone as fo
import fiftyone.zoo as foz
import json
import re
import time
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class MultimodalityPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="multimodality_panel",
            label="Multimodality VLM Testing",
            description="Test multiple VLMs dynamically with metrics evaluation",
            dynamic=True,
        )

    def load_panel_data(self, ctx):
        """Load and cache panel data"""
        if hasattr(self, "_cache"):
            return self._cache
        
        current_dataset = ctx.dataset
        current_view = ctx.view

        available_fields = []
        if current_dataset:
            try:
                field_names = []
                for field_name, field in current_dataset.get_field_schema().items():
                    if field_name not in ['id', 'filepath', 'tags', 'metadata']:
                        field_names.append(field_name)
                available_fields = field_names
            except Exception as e:
                logger.error(f"Error loading dataset fields: {e}")
                available_fields = []
        
        available_models = [
            {
                "name": "fastvlm", 
                "label": "FastVLM-1.5B", 
                "type": "zoo",
                "description": "Apple's fast vision-language model",
                "size": "1.5B parameters",
                "speed": "Fast",
                "accuracy": "Good"
            },
            {
                "name": "gemini", 
                "label": "Gemini Vision", 
                "type": "api",
                "description": "Google Gemini multimodal vision (requires GEMINI_API_KEY)",
                "size": "N/A",
                "speed": "Fast",
                "accuracy": "Excellent"
            },
            {
                "name": "qwen_3b", 
                "label": "Qwen2.5-VL-3B", 
                "type": "zoo",
                "description": "Alibaba's vision-language model",
                "size": "3B parameters",
                "speed": "Medium", 
                "accuracy": "Very Good"
            }
        ]
        
        self._cache = {
            "current_dataset": current_dataset,
            "current_view": current_view,
            "available_fields": available_fields,
            "available_models": available_models,
            "results": ctx.panel.get_state("results", {}),
            "metrics": ctx.panel.get_state("metrics", {}),
            "dataset_name": current_dataset.name if current_dataset else "No Dataset",
            "total_samples": len(current_view) if current_view else (len(current_dataset) if current_dataset else 0)
        }
        
        return self._cache

    def on_load(self, ctx, init=True):
        """Initialize panel state"""
        ctx.panel.state.set("page", 1)
        ctx.panel.state.set("results", {})
        ctx.panel.state.set("metrics", {})
        ctx.panel.state.set("is_running", False)
        self._update(ctx)
    
    def on_change_ctx(self, ctx):
        """Handle context changes"""
        self._update(ctx)



    def update_prompt_from_preset(self, ctx):
        """Update text prompt based on selected preset"""
        preset_name = ctx.panel.get_state("prompt_preset", "driving_scene")
        
        prompt_presets = [
            {
                "name": "driving_scene",
                "label": "Driving Scene Analysis",
                "prompt": """You are given a driving scene image and a proposed driving action. 
                    Based on what you see in the image, determine whether the action is appropriate for the situation. 
                    Answer only in JSON.

                    Format:
                    {{
                    "action": "{}",
                    "judgment": "appropriate" or "not_appropriate",
                    "confidence": 0.0 to 1.0,
                    "reason": "<short explanation>"
                }}"""
            },
            {
                "name": "object_detection",
                "label": "Object Detection",
                "prompt": "Does this image contain a '{}'?"
            },
            {
                "name": "scene_classification",
                "label": "Scene Classification",
                "prompt": "What type of scene is this? The answer should be '{}'."
            },
            {
                "name": "custom",
                "label": "Custom Prompt",
                "prompt": "Analyze this image and determine if the action '{}' is appropriate."
            }
        ]
        
        selected_preset = next((p for p in prompt_presets if p["name"] == preset_name), prompt_presets[0])
        
        ctx.panel.state.set("text_prompt", selected_preset["prompt"])
        
        return {"success": True, "prompt": selected_preset["prompt"]}


    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """Validate text input for dynamic variables with enhanced VLM-specific validation"""
        if not text or not isinstance(text, str):
            return {"valid": False, "error": "Text input is required"}
        
        if len(text.strip()) < 10:
            return {"valid": False, "error": "Text prompt should be at least 10 characters long"}
        
        weird_symbols = re.findall(r'[^\w\s{}.,!?;:\'"()-/\\@#$%^&*+=<>|~`]', text)
        if weird_symbols:
            return {"valid": False, "error": f"Text contains invalid symbols: {set(weird_symbols)}"}
        
        if '{}' not in text and '{' not in text and '}' not in text:
            return {"valid": False, "error": "Text must contain {} for dynamic variables"}
        
        open_brackets = text.count('{')
        close_brackets = text.count('}')
        if open_brackets != close_brackets:
            return {"valid": False, "error": "Unbalanced {} brackets"}
        
        placeholder_count = text.count('{}')
        if placeholder_count > 3:
            return {"valid": False, "error": "Too many {} placeholders (max 3 recommended)"}
        
        return {"valid": True, "error": None, "placeholder_count": placeholder_count}

    def _update(self, ctx):
        """Update panel state (like decompose_core_panel)"""
        if hasattr(self, "_cache"):
            delattr(self, "_cache")
        
        self.load_panel_data(ctx)
        cache = self._cache
        
        ctx.panel.state.set("dataset_name", cache["dataset_name"])
        ctx.panel.state.set("total_samples", cache["total_samples"])
        ctx.panel.state.set("available_fields", cache["available_fields"])
        ctx.panel.state.set("available_models", cache["available_models"])
        
        self.update_prompt_from_preset(ctx)
        
        current_model = ctx.panel.get_state("selected_model")
        last_model = ctx.panel.state.get("last_selected_model")
        if current_model != last_model:
            ctx.panel.state.set("analysis_complete", False)
            ctx.panel.state.set("last_selected_model", current_model)
            ctx.panel.state.set("error_message", None)

    def render(self, ctx):
        """Render the panel UI"""
        view = GridView(align_x="center", align_y="center", orientation="vertical", height=100, width=100, gap=2)
        panel = Object()
        
        dataset_name = ctx.panel.state.get("dataset_name", "No Dataset")
        total_samples = ctx.panel.state.get("total_samples", 0)
        analysis_complete = ctx.panel.state.get("analysis_complete", False)
        is_running = ctx.panel.state.get("is_running", False)
        execution_time = ctx.panel.state.get("execution_time", 0)
        model_tested = ctx.panel.state.get("model_tested", "")
        
        panel.md(
            f"""
            #### Vision-Language Model (VLM) Suite
            
            **Compare multiple VLMs side-by-side** with comprehensive metrics and analysis.
            
            **Current Dataset:** `{dataset_name}` ({total_samples} samples)
            
            """,
            name="intro"
        )
        
        available_fields = ctx.panel.state.get("available_fields", [])
        panel.enum(
            "selected_field",
            available_fields,
            default=available_fields[0] if available_fields else None,
            label="Prompt Field (Mandatory)",
            description="Choose a field to use in the prompt template"
        )
        
        panel.enum(
            "ground_truth_field",
            available_fields,
            default=available_fields[0] if available_fields else None,
            label="Ground Truth Field (Mandatory)",
            description="Choose a field containing ground truth labels for evaluation"
        )
        
        prompt_presets = [
            {
                "name": "driving_scene",
                "label": "Driving Scene Analysis",
                "prompt": """You are given a driving scene image and a proposed driving action. 
                    Based on what you see in the image, determine whether the action is appropriate for the situation. 
                    Answer only in JSON.

                    Format:
                    {{
                    "action": "{}",
                    "judgment": "appropriate" or "not_appropriate",
                    "confidence": 0.0 to 1.0,
                    "reason": "<short explanation>"
                    }}
                """
            },
            {
                "name": "object_detection",
                "label": "Object Detection",
                "prompt": "Does this image contain a '{}'?"
            },
            {
                "name": "scene_classification",
                "label": "Scene Classification",
                "prompt": "What type of scene is this? The answer should be '{}'."
            },
            {
                "name": "custom",
                "label": "Custom Prompt",
                "prompt": "Analyze this image and determine if the action '{}' is appropriate."
            }
        ]
        
        preset_choices = [preset["name"] for preset in prompt_presets]
        panel.enum(
            "prompt_preset",
            preset_choices,
            default="driving_scene",
            label="Prompt Template",
            description="Choose a predefined prompt template"
        )
        
        current_prompt = ctx.panel.get_state("text_prompt", prompt_presets[0]["prompt"])
        
        panel.str(
            "text_prompt",
            default=current_prompt,
            label="Text Prompt",
            description="Enter text with {} for dynamic variables (updates when preset changes)"
        )
        
        available_models = ctx.panel.state.get("available_models", [])
        model_choices = [model["name"] for model in available_models]
        
        panel.enum(
            "selected_model",
            model_choices,
            default="fastvlm" if "fastvlm" in model_choices else (model_choices[0] if model_choices else None),
            label="Select Model",
            description="Choose one VLM to test"
        )
        
        error_message = ctx.panel.state.get("error_message")
        if error_message:
            panel.md(
                f"""
                **Error:**
                {error_message}
                """,
                name="error_message"
            )
 
        if is_running:
            panel.md(
                """
                **Running…**
                
                The job is executing. You can monitor detailed progress in the Operators drawer.
                """,
                name="running_status"
            )
        
        if analysis_complete and not is_running:
            results = ctx.panel.state.get("results", {})
            
            panel.md(
                f"""
                **Analysis Complete!**
                
                **Execution Summary:**
                - **Model Tested:** {model_tested}
                - **Total Time:** {execution_time:.2f} seconds
                
                **Model Results:**
                """,
                name="completion_status"
            )
            
            for model_name, result in results.items():
                if result.get("success"):
                    panel.md(
                        f"""
                        **{result.get('model', model_name)}**
                        - **Execution Time:** {result.get('execution_time', 0):.2f}s
                        - **Output Field:** `{result.get('output_field', 'unknown')}`
                        - **Status:** {result.get('message', 'Completed')}
                        """,
                        name=f"model_result_{model_name}"
                    )
                else:
                    panel.md(
                        f"""
                        **{model_name}**
                        - **Error:** {result.get('error', 'Unknown error')}
                        - **Description:** {result.get('description', 'No description')}
                        """,
                        name=f"model_error_{model_name}"
                    )
            
            panel.md(
                """
                **Next Steps:**
                1. **Check Results** → Look for new fields in your dataset (e.g., `fastvlm_results`, `qwen_3b_results`)
                2. **Use Evaluation Panel** → Go to FiftyOne's evaluation panel to compare model performance
                3. **Filter Results** → Use FiftyOne's filtering to analyze specific subsets
                4. **Export Data** → Export results for further analysis
                
                **Pro Tip:** Use FiftyOne's built-in evaluation tools to get detailed metrics and visualizations!
                """,
                name="next_steps"
            )
        elif not is_running:
            panel.md(
                """
                **Ready to Run Analysis?**
                
                Click the button below to execute the selected model on your current view.
                Results will be stored in your dataset for evaluation.
                """,
                name="run_instruction"
            )
            
            # Bottom action button to trigger operator or run flow
            panel.btn(
                "run_operator_btn",
                label="Run VLM Analysis",
                on_click=self._on_click_run,
                variant="contained",
            )
        
        return Property(panel, view=view)


    def _on_click_run(self, ctx):
        """Handle Run button: validate inputs then execute operator-based pipeline"""
        try:
            selected_model = ctx.panel.get_state("selected_model")
            selected_field = ctx.panel.get_state("selected_field")
            ground_truth_field = ctx.panel.get_state("ground_truth_field")
            text_prompt = ctx.panel.get_state("text_prompt")

            # Panel-side validation so the operator only runs on valid inputs
            if not selected_field:
                ctx.panel.state.set("error_message", "Please select a prompt field")
                ctx.panel.state.set("analysis_complete", False)
                return

            if not ground_truth_field:
                ctx.panel.state.set("error_message", "Please select a ground truth field")
                ctx.panel.state.set("analysis_complete", False)
                return

            if not text_prompt:
                ctx.panel.state.set("error_message", "Please enter a text prompt")
                ctx.panel.state.set("analysis_complete", False)
                return

            validation = self.validate_text_input(text_prompt)
            if not validation.get("valid"):
                ctx.panel.state.set("error_message", validation.get("error", "Invalid prompt"))
                ctx.panel.state.set("analysis_complete", False)
                return

            if not selected_model:
                ctx.panel.state.set("error_message", "Please select a model")
                ctx.panel.state.set("analysis_complete", False)
                return

            ctx.panel.state.set("error_message", None)

            params = {
                "selected_model": selected_model,
                "selected_field": selected_field,
                "ground_truth_field": ground_truth_field,
                "text_prompt": text_prompt,
            }

            ctx.panel.state.set("is_running", True)
            context = {
                "view": ctx.view,
                "params": params,
            }
            execution_result = foo.execute_operator("@adonaivera/fiftyone-agents/vlm_pipeline_operator", context)

            if execution_result and hasattr(execution_result, 'result'):
                result = execution_result.result
                if result and hasattr(result, 'success') and result.success:
                    ctx.panel.state.set("execution_time", getattr(result, 'execution_time', 0))
                    ctx.panel.state.set("timestamp", getattr(result, 'timestamp', None))
                    ctx.panel.state.set("model_tested", getattr(result, 'model', ''))
                    ctx.panel.state.set("results", getattr(result, 'results', {}))
                    ctx.panel.state.set("error_message", None)
                    ctx.panel.state.set("analysis_complete", True)
                else:
                    error_msg = getattr(result, 'error', 'Unknown error') if result else "No result returned"
                    ctx.panel.state.set("error_message", str(error_msg))
                    ctx.panel.state.set("analysis_complete", False)
            else:
                ctx.panel.state.set("error_message", "Failed to execute operator")
                ctx.panel.state.set("analysis_complete", False)
        except Exception as e:
            logger.error(f"Error in _on_click_run: {e}")
            ctx.panel.state.set("error_message", str(e))
            ctx.panel.state.set("analysis_complete", False)
        finally:
            ctx.panel.state.set("is_running", False)


def register(p):
    p.register(MultimodalityPanel)

