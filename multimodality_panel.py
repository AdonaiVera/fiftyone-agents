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
                "name": "fastvlm_7b", 
                "label": "FastVLM-7B", 
                "type": "zoo",
                "description": "Apple's larger vision-language model",
                "size": "7B parameters", 
                "speed": "Medium",
                "accuracy": "Very Good"
            },
            {
                "name": "openai", 
                "label": "OpenAI GPT-4V", 
                "type": "api",
                "description": "OpenAI's vision model (requires API key)",
                "size": "Unknown",
                "speed": "Slow",
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
            },
            {
                "name": "qwen_7b", 
                "label": "Qwen2.5-VL-7B", 
                "type": "zoo",
                "description": "Alibaba's larger vision-language model",
                "size": "7B parameters",
                "speed": "Slow",
                "accuracy": "Excellent"
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

    def run_vlm_analysis(self, ctx):
        """Run VLM analysis using existing FiftyOne plugins"""
        try:
            selected_field = ctx.panel.get_state("selected_field")
            text_prompt = ctx.panel.get_state("text_prompt")
            selected_model = ctx.panel.get_state("selected_model")
            
            current_view = ctx.view
            if not current_view:
                return {"error": "No dataset/view available. Please select a dataset in FiftyOne."}
            
            if not selected_field:
                return {"error": "Please select a field"}
            
            if not text_prompt:
                return {"error": "Please enter a text prompt"}
            
            validation = self.validate_text_input(text_prompt)
            if not validation["valid"]:
                return {"error": validation["error"]}
            
            if not selected_model:
                return {"error": "Please select a model"}
            
            dataset = current_view
            
            field_values = dataset.values(selected_field)
            dynamic_prompts = []
            
            for value in field_values:
                if value is not None:
                    dynamic_prompt = text_prompt.replace('{}', str(value))
                    dynamic_prompts.append(dynamic_prompt)
                else:
                    dynamic_prompts.append(text_prompt)
            
            dataset.set_values("dynamic_prompt", dynamic_prompts)
            
            model_start_time = time.time()
            results = {}
            
            try:
                logger.info(f"Starting model {selected_model}")
                
                if selected_model == "fastvlm":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-1.5B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field="fastvlm_results")
                    result = {"success": True, "model": "FastVLM-1.5B", "output_field": "fastvlm_results"}
                    
                elif selected_model == "fastvlm_7b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-7B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field="fastvlm_7b_results")
                    result = {"success": True, "model": "FastVLM-7B", "output_field": "fastvlm_7b_results"}
                    
                elif selected_model == "openai":
                    try:
                        operator_result = ctx.ops.execute_operator(
                            "@jacobmarks/gpt4_vision/query_gpt4_vision",
                            {
                                "query_text": text_prompt,
                                "max_tokens": 1000
                            }
                        )
                        
                        if operator_result and operator_result.get("success"):
                            result = {"success": True, "model": "OpenAI GPT-4V", "output_field": "gpt4_vision_results"}
                        else:
                            error_msg = operator_result.get("error", "Unknown error") if operator_result else "No result returned"
                            result = {"error": f"GPT-4 Vision execution failed: {error_msg}"}
                            
                    except Exception as e:
                        logger.error(f"GPT-4 Vision plugin error: {e}")
                        result = {"error": f"GPT-4 Vision plugin error: {str(e)}"}
                    
                elif selected_model == "qwen_3b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-3B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field="qwen_3b_results")
                    result = {"success": True, "model": "Qwen2.5-VL-3B", "output_field": "qwen_3b_results"}
                    
                elif selected_model == "qwen_7b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-7B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field="qwen_7b_results")
                    result = {"success": True, "model": "Qwen2.5-VL-7B", "output_field": "qwen_7b_results"}
                
                else:
                    result = {"error": f"Unknown model: {selected_model}"}
                
                model_time = time.time() - model_start_time
                
                if result.get("success"):
                    logger.info(f"Completed model {selected_model} in {model_time:.2f} seconds")
                    result["execution_time"] = model_time
                    results[selected_model] = result
                else:
                    logger.error(f"Model {selected_model} failed: {result.get('error', 'Unknown error')}")
                    results[selected_model] = result
                
            except Exception as e:
                logger.error(f"Error running model {selected_model}: {e}")
                results[selected_model] = {"error": str(e)}
            
            total_time = time.time() - model_start_time
            logger.info(f"Model completed in {total_time:.2f} seconds")
            
            ctx.panel.state.set("execution_time", total_time)
            ctx.panel.state.set("timestamp", datetime.now().isoformat())
            ctx.panel.state.set("model_tested", selected_model)
            ctx.panel.state.set("analysis_complete", True)
            ctx.panel.state.set("results", results)
            
            ctx.ops.reload_dataset()
            
            return {
                "success": True, 
                "message": f"Successfully ran {selected_model} in {total_time:.2f}s",
                "model": selected_model,
                "execution_time": total_time,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in run_vlm_analysis: {e}")
            return {"error": f"Failed to run models: {str(e)}"}


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
        
        run_analysis = ctx.panel.get_state("run_analysis", False)
        if run_analysis and not ctx.panel.state.get("analysis_complete", False):
            ctx.panel.state.set("run_analysis", False)
            ctx.panel.state.set("is_running", True)
            
            result = self.run_vlm_analysis(ctx)
            
            if result.get("error"):
                ctx.panel.state.set("error_message", result["error"])
            else:
                ctx.panel.state.set("error_message", None)
                ctx.panel.state.set("analysis_complete", True)
            ctx.panel.state.set("is_running", False)

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
            #### Vision-Language Model (VLM) Testing Suite
            
            **Compare multiple VLMs side-by-side** with comprehensive metrics and analysis.
            
            **Current Dataset:** `{dataset_name}` ({total_samples} samples)
            
            **Supported Models (via FiftyOne Plugins):**
            - **FastVLM** (1.5B & 7B) - Apple's efficient models via [FastVLM plugin](https://github.com/harpreetsahota204/fast_vlm)
            - **Qwen2.5-VL** (3B & 7B) - Alibaba's powerful models via [Qwen2.5-VL plugin](https://github.com/harpreetsahota204/qwen2_5_vl)
            - **OpenAI GPT-4V** - Industry-leading accuracy via [GPT-4 Vision plugin](https://github.com/jacobmarks/gpt4-vision-plugin)
            
            **Quick Setup:**
            1. **Install Plugins** → Install required plugins (see links above)
            2. **Set API Keys** → Set OPENAI_API_KEY environment variable for GPT-4V
            3. **Pick Field** → Ground truth field for evaluation
            4. **Choose Template** → Select from predefined prompts
            5. **Customize Prompt** → Modify template or write custom
            6. **Choose Model** → Select one VLM to test
            7. **Run Analysis** → Execute model and store results
            8. **Check Evaluation** → Use FiftyOne's evaluation panel
            
            **Available Templates:**
            - **Driving Scene Analysis** → JSON-formatted action evaluation
            - **Object Detection** → Simple object presence check
            - **Scene Classification** → Scene type identification
            - **Custom Prompt** → Write your own template
            """,
            name="intro"
        )
        
        available_fields = ctx.panel.state.get("available_fields", [])
        panel.enum(
            "selected_field",
            available_fields,
            default=available_fields[0] if available_fields else None,
            label="Select Field (Mandatory)",
            description="Choose a field from the selected dataset"
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
        
        model_info_data = []
        for model in available_models:
            model_info_data.append({
                "name": model["name"],
                "label": model["label"],
                "size": model["size"],
                "speed": model["speed"],
                "accuracy": model["accuracy"],
                "description": model["description"]
            })
        
        model_info_table = TableView()
        model_info_table.add_column("label", label="Model")
        model_info_table.add_column("size", label="Size")
        model_info_table.add_column("speed", label="Speed")
        model_info_table.add_column("accuracy", label="Accuracy")
        model_info_table.add_column("description", label="Description")
        
        panel.list("model_info_table", TypeObject(), view=model_info_table, label="Available Models")
        ctx.panel.state.set("model_info_table", model_info_data)
        
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
            
            panel.bool(
                "run_analysis",
                default=False,
                label="Run VLM Analysis",
                description="Click to execute selected model and store results in dataset"
            )
        
        return Property(panel, view=view)


def register(p):
    p.register(MultimodalityPanel)

