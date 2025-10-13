import fiftyone.operators as foo
import fiftyone.zoo as foz
import fiftyone as fo
import time
import json
import re

def parse_vlm_response_to_classification(response_text, model_name, ground_truth_value):
    """
    Parse VLM JSON response and convert to FiftyOne classification format.
    
    Expected format: {"action": "forward", "judgment": "appropriate", "reason": "..."}
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            action = parsed.get("action", "unknown")
            judgment = parsed.get("judgment", "unknown")
            reason = parsed.get("reason", "")
            
            label = action if judgment == "appropriate" else ground_truth_value
                
            classification = fo.Classification(
                label=label,
                confidence=1.0
            )
            
            classification.tags = [
                f"action_{action}",
                f"judgment_{judgment}",
                f"model_{model_name}",
                f"ground_truth_{ground_truth_value}"
            ]            
            return classification
            
    except Exception:
        classification = fo.Classification(label="unknown", confidence=1.0)
        classification.tags = [f"model_{model_name}", "parsing_error"]
        return classification


class RunVLMPipeline(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlm_pipeline_operator",
            label="Run VLM Pipeline",
            dynamic=True,
        )

    def execute(self, ctx):
        try:
            selected_field = ctx.params.get("selected_field")
            text_prompt = ctx.params.get("text_prompt")
            selected_model = ctx.params.get("selected_model")
            ground_truth_field = ctx.params.get("ground_truth_field")

            current_view = ctx.view
            if not current_view:
                return {"error": "No dataset/view available. Please select a dataset in FiftyOne."}

            dataset = current_view

            field_values = dataset.values(selected_field)
            dynamic_prompts = []
            for value in field_values:
                if value is not None:
                    dynamic_prompts.append(text_prompt.replace('{}', str(value)))
                else:
                    dynamic_prompts.append(text_prompt)
            dataset.set_values("dynamic_prompt", dynamic_prompts)

            model_start_time = time.time()
            results = {}

            try:
                raw_output_field = f"{selected_model}_raw_results"
                classification_field = f"{selected_model}_predictions"
                
                if selected_model == "fastvlm":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-1.5B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field)
                    result = {"success": True, "model": "FastVLM-1.5B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "fastvlm_7b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-7B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field)
                    result = {"success": True, "model": "FastVLM-7B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "openai":
                    operator_result = foo.execute_operator(
                        "@jacobmarks/gpt4_vision/query_gpt4_vision", ctx, params={"query_text": text_prompt, "max_tokens": 1000}
                    )
                    if operator_result and operator_result.get("success"):
                        result = {"success": True, "model": "OpenAI GPT-4V", "raw_field": "gpt4_vision_results", "classification_field": classification_field}
                    else:
                        error_msg = operator_result.get("error", "Unknown error") if operator_result else "No result returned"
                        result = {"error": f"GPT-4 Vision execution failed: {error_msg}"}

                elif selected_model == "qwen_3b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-3B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field)
                    result = {"success": True, "model": "Qwen2.5-VL-3B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "qwen_7b":
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-7B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field)
                    result = {"success": True, "model": "Qwen2.5-VL-7B", "raw_field": raw_output_field, "classification_field": classification_field}

                else:
                    result = {"error": f"Unknown model: {selected_model}"}

                model_time = time.time() - model_start_time
                if result.get("success"):
                    result["execution_time"] = model_time
                    
                    # Convert raw VLM responses to classifications
                    raw_field = result["raw_field"]
                    classification_field = result["classification_field"]
                    
                    # Process each sample to convert raw response to classification
                    classifications = []
                    for sample in dataset:
                        raw_response = sample.get_field(raw_field)
                        ground_truth_value = sample.get_field(ground_truth_field)
                        
                        if raw_response:
                            if hasattr(raw_response, 'response'):
                                response_text = raw_response.response
                            elif hasattr(raw_response, 'text'):
                                response_text = raw_response.text
                            else:
                                response_text = str(raw_response)
                            
                            classification = parse_vlm_response_to_classification(
                                response_text, selected_model, ground_truth_value
                            )
                            classifications.append(classification)
                        else:
                            classifications.append(None)
                    
                    dataset.set_values(classification_field, classifications)                    
                    if ground_truth_field in dataset.get_field_schema():
                        try:
                            evaluation_key = f"{selected_model}_evaluation"
                            dataset.evaluate_classifications(
                                pred_field=classification_field,
                                gt_field=ground_truth_field,
                                eval_key=evaluation_key
                            )
                            result["evaluation_key"] = evaluation_key
                        except Exception as eval_error:
                            result["evaluation_error"] = str(eval_error)
                    
                    results[selected_model] = result
                else:
                    results[selected_model] = result

            except Exception as e:
                results[selected_model] = {"error": str(e)}

            total_time = time.time() - model_start_time
            ctx.ops.reload_dataset()

            return {
                "success": True,
                "message": f"Successfully ran {selected_model} in {total_time:.2f}s",
                "model": selected_model,
                "execution_time": total_time,
                "results": results,
            }

        except Exception as e:
            return {"error": f"Failed to run models: {str(e)}"}



def register(p):
    p.register(RunVLMPipeline)
