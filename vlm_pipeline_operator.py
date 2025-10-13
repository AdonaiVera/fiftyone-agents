import fiftyone.operators as foo
import fiftyone.zoo as foz
import fiftyone as fo
import time
import json
import re
import torch
import gc

def cleanup_memory():
    """Clean up GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def parse_vlm_response_to_classification(response_text, model_name, ground_truth_value):
    """
    Parse VLM JSON response and convert to FiftyOne classification format.
    
    Expected format: {"action": "forward", "judgment": "appropriate", "confidence": 0.8, "reason": "..."}
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            action = parsed.get("action", "unknown")
            judgment = parsed.get("judgment", "unknown")
            confidence = parsed.get("confidence", 1.0)
            reason = parsed.get("reason", "")

            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))  
            except (ValueError, TypeError):
                confidence = 1.0
            
            label = action if judgment == "appropriate" else ground_truth_value
                
            classification = fo.Classification(
                label=label,
                confidence=confidence
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
                    if hasattr(value, 'label'):
                        label_value = value.label
                    elif hasattr(value, 'classifications') and value.classifications:
                        label_value = value.classifications[0].label
                    else:
                        label_value = str(value)
                    dynamic_prompts.append(text_prompt.replace('{}', label_value))
                else:
                    dynamic_prompts.append(text_prompt)
            dataset.set_values("dynamic_prompt", dynamic_prompts)

            model_start_time = time.time()
            results = {}

            try:
                raw_output_field = f"{selected_model}_raw_results"
                classification_field = f"{selected_model}_predictions"
                
                if selected_model == "fastvlm":
                    cleanup_memory() 
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-1.5B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field, image_field="filepath")
 
                    result = {"success": True, "model": "FastVLM-1.5B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "fastvlm_7b":
                    cleanup_memory()  
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/fast_vlm")
                    model = foz.load_zoo_model("apple/FastVLM-7B")
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field, image_field="filepath")
                    result = {"success": True, "model": "FastVLM-7B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "openai":
                    raw_output_field = f"{selected_model}_raw_results"
                    underlying_dataset = dataset._dataset if hasattr(dataset, '_dataset') else dataset
                    underlying_dataset.add_sample_field(raw_output_field, fo.StringField, overwrite=True)
                    
                    # Create context object that supports attribute access (required by GPT-4 plugin)
                    class ContextObject:
                        def __init__(self, view, dataset, selected, params):
                            self.view = view
                            self.dataset = dataset
                            self.selected = selected
                            self.params = params
                    
                    for sample in dataset:
                        # Create a proper context with the sample's image
                        single_sample_view = dataset.select(sample.id)
                        
                        single_ctx = ContextObject(
                            view=single_sample_view,
                            dataset=underlying_dataset,
                            selected=[sample.id],  
                            params={"query_text": text_prompt, "max_tokens": 1000}
                        )
                        
                        try:
                            operator_result = foo.execute_operator(
                                "@jacobmarks/gpt4_vision/query_gpt4_vision", context=single_ctx, params={"query_text": text_prompt, "max_tokens": 1000}
                            )
                            if operator_result and hasattr(operator_result, 'result') and operator_result.result:
                                result_dict = operator_result.result
                                if isinstance(result_dict, dict) and 'answer' in result_dict:
                                    response_text = result_dict['answer']
                                    if "Incorrect API key provided" in response_text or "API key" in response_text:
                                        raise Exception(f"OpenAI API Key Error: {response_text}")
                                    sample[raw_output_field] = response_text
                                else:
                                    response_text = str(result_dict)
                                    sample[raw_output_field] = response_text
                            else:
                                sample[raw_output_field] = None
                        except Exception as e:
                            sample[raw_output_field] = None
                            if "OpenAI API Key Error" in str(e):
                                raise e
                        sample.save()
                    
                    try:
                        result = {"success": True, "model": "OpenAI GPT-4V", "raw_field": raw_output_field, "classification_field": classification_field}
                    except Exception as e:
                        if "OpenAI API Key Error" in str(e):
                            result = {"error": str(e)}
                        else:
                            result = {"error": f"OpenAI execution failed: {str(e)}"}
                    

                elif selected_model == "qwen_3b":
                    cleanup_memory()  
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-3B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field, image_field="filepath")
                    result = {"success": True, "model": "Qwen2.5-VL-3B", "raw_field": raw_output_field, "classification_field": classification_field}

                elif selected_model == "qwen_7b":
                    cleanup_memory() 
                    foz.register_zoo_model_source("https://github.com/harpreetsahota204/qwen2_5_vl")
                    model = foz.load_zoo_model("Qwen/Qwen2.5-VL-7B-Instruct")
                    model.operation = "vqa"
                    dataset.apply_model(model, prompt_field="dynamic_prompt", label_field=raw_output_field, image_field="filepath")

                    result = {"success": True, "model": "Qwen2.5-VL-7B", "raw_field": raw_output_field, "classification_field": classification_field}

                else:
                    result = {"error": f"Unknown model: {selected_model}"}

                model_time = time.time() - model_start_time
                if result.get("success"):
                    result["execution_time"] = model_time
                    
                    # Convert raw VLM responses to classifications
                    raw_field = result["raw_field"]
                    classification_field = result["classification_field"]
                    
                    # First, create eval_gt_field with first classifications for evaluation
                    eval_gt_field = f"{ground_truth_field}_first"
                    first_classifications = []
                    for sample in dataset:
                        gt_value = sample[ground_truth_field]
                        if gt_value and hasattr(gt_value, 'classifications') and gt_value.classifications:
                            first_classifications.append(gt_value.classifications[0])
                        else:
                            first_classifications.append(gt_value)
                    dataset.set_values(eval_gt_field, first_classifications)
                    
                    # Process each sample to convert raw response to classification
                    classifications = []
                    for sample in dataset:
                        raw_response = sample.get_field(raw_field)
                        ground_truth_raw = sample.get_field(eval_gt_field)
                        ground_truth_value = ground_truth_raw.label if ground_truth_raw and hasattr(ground_truth_raw, 'label') else None  
                        
                        if raw_response:
                            if hasattr(raw_response, 'response'):
                                response_text = raw_response.response
                            elif hasattr(raw_response, 'text'):
                                response_text = raw_response.text
                            elif isinstance(raw_response, str):
                                response_text = raw_response
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
                                gt_field=eval_gt_field,
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
