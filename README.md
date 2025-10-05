# 🤖 FiftyOne VLM Testing Suite

A comprehensive **FiftyOne plugin** for testing and evaluating multiple **Vision-Language Models (VLMs)** with dynamic prompts and built-in evaluation capabilities.

## What This Plugin Offers

### Panel: `Multimodality VLM Testing`

An interactive panel interface for comprehensive VLM testing and evaluation with **dynamic view-based analysis**.

#### **Key Capabilities**

- **Dynamic View Integration**: Automatically responds to your current FiftyOne view
- **Dynamic Prompting**: Create prompts with field substitutions using `{field_name}` syntax
- **Multi-Model Support**: Test FastVLM, OpenAI GPT-4V, and Qwen2.5-VL models
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

**OpenAI GPT-4V**:
- **GPT-4 Vision** - Industry-leading accuracy and reasoning
- *Via [GPT-4 Vision plugin](https://github.com/jacobmarks/gpt4-vision-plugin)*

#### **Features**

- **Prompt Templates**: Pre-defined templates for common VLM tasks
- **Dynamic Field Substitution**: Use `{field_name}` syntax to inject ground truth data
- **Single Model Testing**: Focus on one model at a time for detailed analysis
- **Results Storage**: Automatically store VLM outputs in your dataset

## Installation

### 1. Install Required Plugins

```bash
# FastVLM Plugin
fiftyone plugins download https://github.com/harpreetsahota204/fast_vlm

# Qwen2.5-VL Plugin
fiftyone plugins download https://github.com/harpreetsahota204/qwen2_5_vl

# GPT-4 Vision Plugin
fiftyone plugins download https://github.com/jacobmarks/gpt4-vision-plugin
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Download This Plugin

```bash
fiftyone plugins download https://github.com/AdonaiVera/fiftyone-agents
```

## How to Use

### **Step 1: Prepare Your Dataset**

```python
import fiftyone as fo

# Load your dataset
dataset = fo.Dataset("my_vlm_dataset")
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
- **GPT-4V**: Highest accuracy for complex reasoning tasks (no local memory usage)

### **Memory Management**
- Start with smaller models if you have limited RAM
- Use filtered views to test on smaller subsets first
- Monitor memory usage during model execution
- Consider using OpenAI GPT-4V for memory-constrained environments

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
