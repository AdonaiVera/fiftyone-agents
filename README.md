# 🤖 FiftyOne VLM Testing Suite

A comprehensive **FiftyOne plugin** for testing and evaluating multiple **Vision-Language Models (VLMs)** with dynamic prompts and built-in evaluation capabilities.

![Screencastfrom2025-10-0500-19-05-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/81d8a15d-ff3e-425b-9f6e-21b4da179c1b)

## What This Plugin Offers

### Panel: `Multimodality VLM Testing`

An interactive panel interface for comprehensive VLM testing and evaluation with **dynamic view-based analysis**.

#### **Key Capabilities**

- **Dynamic View Integration**: Automatically responds to your current FiftyOne view
- **Dynamic Prompting**: Create prompts with field substitutions using `{field_name}` syntax
- **Multi-Model Support**: Test FastVLM, Gemini Vision, and Qwen2.5-VL models
- **Built-in Evaluation**: Leverage FiftyOne's evaluation panel for comprehensive metrics

#### **Supported Models**

**FastVLM Models**:
- **FastVLM-1.5B** - Apple's efficient 1.5B parameter model
- **FastVLM-7B** - Apple's powerful 7B parameter model
- *Via [FastVLM plugin](https://github.com/harpreetsahota204/fast_vlm)*

**Qwen2.5-VL Models**:
- **Qwen2.5-VL-3B** - Alibaba's efficient 3B parameter model
- **Qwen2.5-VL-7B** - Alibaba's powerful 7B parameter model
- *Via [Qwen2.5-VL plugin](https://github.com/harpreetsahota204/qwen2_5_vl)*

**Gemini Vision Models**:
- **Gemini-2.5-Flash** - Google's efficient multimodal model
- **Gemini-Pro-Vision** - Google's powerful vision-language model
- *Via [Gemini Vision plugin](https://github.com/AdonaiVera/gemini-vision-plugin)*

#### **Features**

- **Prompt Templates**: Pre-defined templates for common VLM tasks
- **Dynamic Field Substitution**: Use `{field_name}` syntax to inject ground truth data
- **Single Model Testing**: Focus on one model at a time for detailed analysis
- **Results Storage**: Automatically store VLM outputs in your dataset

## Installation

### 1. Download This Plugin

```bash
fiftyone plugins download https://github.com/AdonaiVera/fiftyone-agents
```

### 2. Install Dependencies

```bash
fiftyone plugins requirements @adonaivera/fiftyone-agents --install
```

### 3. Download Required VLM Plugins

After installing the main plugin, you need to download the individual VLM plugins:

```bash
# Download Gemini Vision plugin
fiftyone plugins download https://github.com/AdonaiVera/gemini-vision-plugin

# Download FastVLM plugin
fiftyone plugins download https://github.com/harpreetsahota204/fast_vlm

# Download Qwen2.5-VL plugin
fiftyone plugins download https://github.com/harpreetsahota204/qwen2_5_vl
```

> **Note**: You can also add your own VLM models by following the same pattern and integrating them into the `vlm_pipeline_operator.py` file.

### 4. Set Environment Variables [For API-based models]

```bash
# For Gemini Vision (required)
export GEMINI_API_KEY="your-gemini-api-key-here"
```

## How to Use

### **Step 1: Prepare Your Dataset (your own or the demo)**

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Option A — Load your own dataset (recommended)
# If you already registered a dataset in FiftyOne:
dataset = fo.load_dataset("your-dataset-name")

# Option B — Use the demo dataset (quick start)
# This downloads if needed, then we make it persistent
dataset = foz.load_zoo_dataset(
    "https://github.com/AdonaiVera/bddoia-fiftyone",
    split="validation",
    max_samples=10
)
dataset.persistent = True
```

### **Step 2: Launch FiftyOne and Open the Panel**

```python
session = fo.launch_app(dataset)
session.wait()
```

### **Step 3: Use the VLM Testing Panel**

1. **Open the Panel**: Look for "Multimodality VLM Testing" in your panels
2. **Select Ground Truth Field**: Choose the field containing your ground truth data
3. **Choose Prompt Template**: Select from predefined templates or create custom prompts
4. **Customize Prompt**: Use `{field_name}` syntax to inject dynamic content
5. **Select Model**: Choose one VLM model to test
6. **Run Analysis**: Click "Run VLM Analysis" to execute the model
7. **Check Results**: Use FiftyOne's evaluation panel to analyze performance

## Practical Use Cases

### **Model Comparison**
- Compare different VLMs on the same dataset
- Evaluate which models perform best for specific tasks
- Balance inference speed against accuracy for production use

### **Prompt Engineering**
- Experiment with different prompt structures
- Use dynamic field substitution for contextual prompts
- Test how prompt variations affect model performance

### **Dataset Analysis**
- Identify challenging samples across different models
- Find samples where models disagree or fail
- Test models on filtered views of your data

## Pro Tips

### **Dynamic Prompts**
- Use `{field_name}` to create contextual prompts
- Start with predefined templates and customize as needed
- Include ground truth data in prompts for better evaluation

### **Model Selection**
- **FastVLM**: Best for speed-critical applications and low-memory systems
- **Qwen2.5-VL**: Excellent balance of performance and efficiency
- **Gemini Vision**: High accuracy for complex reasoning tasks with API-based inference

### **Memory Management**
- Start with smaller models if you have limited RAM
- Use filtered views to test on smaller subsets first
- Monitor memory usage during model execution
- Consider using Gemini Vision for memory-constrained environments (API-based)

### **Evaluation**
- Use FiftyOne's evaluation panel for comprehensive analysis
- Test multiple models on the same samples
- Focus on samples where models disagree

## Future Enhancements

- **Batch Model Testing**: Test multiple models simultaneously
- **Custom Model Integration**: Support for additional VLM architectures
- **Advanced Metrics**: More sophisticated evaluation metrics
- **Export Capabilities**: Save results for external analysis

## Credits

* Built with ❤️ on top of FiftyOne by Voxel51
* VLM integrations via community plugins
* Evaluation powered by FiftyOne's built-in evaluation framework

## Contributors

This plugin was developed and maintained by:

* [Adonai Vera](https://github.com/AdonaiVera) 
* [Paula Ramoas](https://github.com/paularamo) 

We welcome more contributors to extend support for additional models, evaluation metrics, and new testing capabilities!
