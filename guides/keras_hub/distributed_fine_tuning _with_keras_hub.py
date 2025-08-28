"""
Title: Distributed Fine-tuning with Keras Hub Models
Author: [Laxmareddy Patlolla](https://github.com/laxmareddyp),[Divyashree Sreepathihalli](https://github.com/divyashreepathihalli)
Date created: 2025/08/28
Last modified: 2025/08/28
Description: Distributed Fine-Tuning with Keras Hub.
Accelerator: GPU
"""

# Use Keras 3 distribution API.
# We'll use Keras 3 native operations where possible
import os
os.environ["KERAS_BACKEND"] = "jax"


import json
import numpy as np
import keras
from keras import Model
import keras_hub
import jax
import tensorflow as tf

# Define model presets for easy access
TEXT_PRESET = "gemma3_instruct_1b"  # Default text model for fine-tuning

"""
## Welcome to Your Distributed Fine-tuning Adventure!

Hey there, AI explorer! Ready to dive into something really exciting? We're about to embark on a journey that will take us from single-device training to harnessing the power of multiple accelerators for lightning-fast model fine-tuning!

**What makes this special?** Instead of waiting hours (or days!) for your model to learn, we'll use distributed training to speed things up dramatically. It's like having a team of workers instead of just one - they can all work on different parts of your data simultaneously!

**What we're going to discover together:**

🎯 **The Magic of Distributed Training**: Learn how to split your workload across multiple GPUs or TPUs
🚀 **Keras 3 Distribution API**: Experience the future of distributed computing with Keras
💎 **Parameter-Efficient Fine-tuning**: Master LoRA (Low-Rank Adaptation) for quick, memory-efficient training
🔄 **Backend Flexibility**: Train on JAX, TensorFlow, or PyTorch backends seamlessly
🎨 **Real-World Applications**: Fine-tune Gemma 3 for your specific use case

**Why This Matters**: Distributed training isn't just about speed - it's about making previously impossible tasks possible. Large language models like Gemma 3 can be fine-tuned in minutes instead of hours, opening up new possibilities for rapid prototyping and experimentation.

Let's get started on this amazing journey! 🌟
"""

"""
## 🔧 Keras 3 Native Operations Strategy

This guide prioritizes Keras 3 native operations for maximum backend compatibility:
- **Primary**: Use `keras.ops.*`, `keras.utils.*`, and `keras.distribution.*`
- **Data Handling**: Use TensorFlow's `tf.data.Dataset` API (compatible with all backends)
- **Goal**: Backend-agnostic code that works with JAX, TensorFlow, and PyTorch
"""

"""
## 🎯 Choose Your Distribution Strategy (Keras 3 Distribution API)

Welcome to the exciting world of distributed training! Today we'll harness the power of multiple devices
to accelerate our Gemma 3 fine-tuning journey.

We'll use the Keras 3 distribution API (currently implemented for the JAX backend), which gives us:

- **Data parallelism**: `keras.distribution.DataParallel` replicates weights across devices
  and shards inputs along the batch dimension - think of it as having multiple workers
  processing different parts of your data simultaneously!
- **Optional model parallelism**: `keras.distribution.ModelParallel` lets you shard large
  model weights across devices using `LayoutMap` - perfect for those massive models that
  won't fit on a single device.

The beauty of this approach is that we'll set a global distribution once, and then all our
models automatically scale across available devices. It's like having a magic wand that
makes your training faster without changing your code!

**Why this matters:**
- Keeps global semantics while scaling across devices - you write code as if you're working
  with one device, but it runs on many!
- Minimal code changes to scale from 1 to many accelerators - just change the distribution
  strategy and watch your training speed up!
"""

# Cell 1: Set up data parallel distribution
def set_data_parallel_distribution():
    """
    🚀 Set up data parallel distribution for maximum training speed!

    This function creates a DataParallel distribution that will automatically:
    - Replicate your model weights across all available devices
    - Split your input data across devices for parallel processing
    - Handle all the complex communication behind the scenes

    It's like having a team of synchronized workers - each one gets a copy of the model
    and processes different parts of your data, then they all share what they learned!
    """
    # Auto-detect local devices. You can also pass a DeviceMesh explicitly.
    dp = keras.distribution.DataParallel()
    keras.distribution.set_distribution(dp)
    print("✅ DataParallel distribution set successfully!")

# Execute this function to set up distributed training
set_data_parallel_distribution()

# Check available devices
def print_num_devices():
    """
    🔍 Let's see how many devices we have available for our distributed training adventure!

    This will show us the total computational power we can harness. More devices = faster training!
    """
    # Only prints device count if JAX is present; otherwise assume 1.
    num = 1
    try:
        num = len(jax.devices())
        print(f"🎉 Found {num} JAX devices for distributed training!")
    except Exception:
        print("ℹ️  JAX not available, assuming single device")
    print(f"🚀 Number of devices available: {num}")
    return num

# Execute this function to see your available devices
num_devices = print_num_devices()

# Enable mixed precision for better performance
def enable_mixed_precision_for_available_accelerator():
    """
    🚀 Enable mixed precision to reduce memory usage and improve throughput!

    This is like switching from high-quality but slow film to fast, efficient digital photography.
    Mixed precision gives us the best of both worlds - speed and accuracy!

    - On GPUs, uses mixed_float16 (FP16 math, FP32 master weights).
    - On TPUs, uses mixed_bfloat16 (BF16 math, safe for TPUs).
    - On CPU only, leaves default (float32).

    Call this before constructing models to ensure variables are created with the
    intended policy.
    """
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")

    # Use Keras 3 native device detection when available
    try:
        gpu_devices = jax.devices("gpu") if jax.devices("gpu") else []
        tpu_devices = jax.devices("tp") if jax.devices("tp") else []
    except:
        # Fallback to basic detection
        gpu_devices = []
        tpu_devices = []

    if backend == "jax":
        # Prefer bfloat16 on JAX when available
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("✅ Enabled mixed_bfloat16 for JAX backend")
    elif gpu_devices:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("✅ Enabled mixed_float16 for GPU training")
    elif tpu_devices:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("✅ Enabled mixed_bfloat16 for TPU training")
    else:
        print("ℹ️  Keeping float32 precision (no GPU/TPU detected)")

# Execute this function to enable mixed precision
enable_mixed_precision_for_available_accelerator()

"""
## 🎯 Load a Gemma 3 Preset (Text-Only for Quick Demos)

Use `keras_hub.models.Gemma3CausalLM.from_preset()` to load a ready-to-fine-tune model.
When loading from a preset, the corresponding preprocessor is attached automatically,
so string inputs can flow directly to `fit()`, `evaluate()`, and `generate()`.

**Why we're using the smallest model:**
- **gemma3_instruct_1b**: Perfect for quick demos and learning - trains fast, fits in memory
- **gemma3_instruct_4b_text**: Great for production text tasks with more capacity

For our distributed training adventure, we'll start with the 1B model to see results quickly,
then you can scale up to larger models once you're comfortable with the workflow!
"""

# Create a Gemma 3 model
def create_model(preset: str = "gemma3_instruct_1b"):
    """
    🎯 Create a Gemma 3 model ready for distributed fine-tuning!

    This function loads a pre-trained Gemma 3 model and prepares it for our training adventure.
    Think of it as getting your car ready for a road trip - we're loading the engine (model),
    checking the fuel (weights), and making sure everything is ready to go!

    **What's happening behind the scenes:**
    - KerasHub downloads the model weights (if not cached)
    - The preprocessor is automatically attached for text processing
    - The model is ready for immediate fine-tuning with `fit()`

    **Why this matters:** This seamless integration means you can focus on training
    instead of worrying about model setup and preprocessing!
    """
    print(f"🚀 Loading Gemma 3 model: {preset}")
    print("⏳ This might take a moment for the first time...")

    model = keras_hub.models.Gemma3CausalLM.from_preset(preset)

    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {model.count_params():,}")
    print(f"🔧 Preprocessor attached: {model.preprocessor is not None}")
    print(f"📝 Preprocessor type: {type(model.preprocessor).__name__}")

    return model

# Execute this function to load your Gemma 3 model
model = create_model("gemma3_instruct_1b")

# Enable LoRA for parameter-efficient fine-tuning
def enable_lora(model: Model, rank: int = 4):
    """
    💎 Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning!

    LoRA is like having a smart adapter that learns to modify your model's behavior
    without changing all the original weights. It's incredibly efficient and powerful!

    **What LoRA does:**
    - Adds small, trainable matrices to key layers
    - Keeps original weights frozen (saves memory)
    - Achieves excellent results with minimal parameters
    - Perfect for quick fine-tuning on new tasks

    **Why rank=4?** This gives us a good balance between:
    - Training speed (faster with smaller rank)
    - Model capacity (enough to learn new patterns)
    - Memory efficiency (fits easily on most devices)

    Think of it as learning 4 new "skills" that can be combined in different ways!
    """
    print(f"💎 Enabling LoRA with rank {rank}...")

    # Enable LoRA on the backbone for PEFT.
    model.backbone.enable_lora(rank=rank)

    # Count trainable parameters
    trainable_params = sum(
        keras.ops.size(w) for w in model.trainable_weights
    )
    total_params = model.count_params()

    print(f"✅ LoRA enabled successfully!")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎯 LoRA efficiency: {trainable_params/total_params*100:.2f}%")

    return model

# Execute this function to enable LoRA
model = enable_lora(model, rank=4)

# Build a text-only dataset
def build_text_only_dataset(batch_size: int = 2):
    """
    📚 Build a text-only dataset for our fine-tuning adventure!

    This creates a simple dataset with question-answer pairs. In real applications,
    you'd load your own data from files, databases, or APIs.

    **What we're creating:**
    - Prompts: Questions or instructions for the model
    - Responses: Desired answers or completions
    - Format: Ready for the Gemma 3 preprocessor

    **Why this matters:** Good data is the foundation of successful fine-tuning.
    The quality and relevance of your training data directly impacts how well
    your model will perform on your specific task!
    """
    features = {
        "prompts": [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "What is 2 + 2?",
            "Write a short poem about AI.",
        ],
        "responses": [
            "The capital of France is Paris.",
            "Machine learning is when computers learn patterns from data to make predictions.",
            "2 + 2 equals 4.",
            "In circuits deep, AI dreams,\nLearning patterns, chasing streams,\nDigital minds that grow and gleam.",
        ],
    }

    # Use TensorFlow dataset API (compatible with all Keras 3 backends)
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.shuffle(16).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"✅ Text dataset created with {len(features['prompts'])} samples")
    print(f"📦 Batch size: {batch_size}")

    return ds

# Execute this function to create your training dataset
train_ds = build_text_only_dataset(batch_size=1)

# Compile the model for training
def compile_model(model: Model):
    """
    ⚙️ Compile the model for our distributed training adventure!

    This is where we configure all the training parameters - think of it as setting up
    the controls for your training journey. We're choosing the optimizer, loss function,
    and generation strategy.

    **What we're configuring:**
    - **Optimizer**: Adam with learning rate 1e-4 (perfect for fine-tuning)
    - **Loss**: "auto" (automatically chooses the right loss for language modeling)
    - **Metrics**: "auto" (automatically selects relevant evaluation metrics)
    - **Sampler**: "greedy" (deterministic text generation for consistent results)

    **Why these choices?** They're proven to work well for language model fine-tuning
    and give us a solid foundation for experimentation!
    """
    print("⚙️ Compiling model for training...")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="auto",
        weighted_metrics="auto",
        sampler="greedy",
    )

    print("✅ Model compiled successfully!")
    print(f"🔧 Optimizer: {type(model.optimizer).__name__}")
    print(f"📊 Learning rate: {model.optimizer.learning_rate}")

    return model

# Execute this function to compile your model
model = compile_model(model)

# Train the model with distributed training
"""
## 🔥 Time to Train with Distributed Power!

Now the magic happens! With our distributed setup, training will automatically scale across
all available devices. Each device will process different parts of your data simultaneously,
dramatically speeding up the learning process.

**What's happening behind the scenes:**
- Your model weights are replicated across all devices
- Input data is automatically split and distributed
- Gradients are synchronized across devices
- All the complex communication is handled automatically

**Why this is exciting:** You're now training with the power of multiple accelerators
without changing a single line of your training code!
"""

# Execute this function to start training
print("🔥 Starting distributed training...")
print("=" * 60)

# Train briefly to see distributed training in action
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/ckpt-{epoch}.keras",
        save_freq="epoch"
    ),
    keras.callbacks.EarlyStopping(monitor="loss", patience=2),
]

# This will automatically use distributed training across all available devices
history = model.fit(
    train_ds,
    epochs=1,
    verbose=2,
    callbacks=callbacks
)

print("✅ Training complete!")

# Test the fine-tuned model
"""
## 🎯 Let's Test What We've Learned!

Now it's time to see the fruits of our distributed training labor! We'll generate some
text using our fine-tuned model to see how well it learned from our training data.

**What we're testing:**
- **Generation quality**: How well does the model respond to prompts?
- **Learning verification**: Did it pick up the patterns from our training data?
- **Distributed inference**: Even generation benefits from our distributed setup!

**Why this matters:** Testing helps us verify that our training was successful and
gives us confidence that our distributed setup is working correctly!
"""

# Execute this function to test your fine-tuned model
print("🎯 Testing the fine-tuned model...")
print("=" * 60)

test_prompts = [
    "What is the capital of France?",
    "Explain machine learning simply.",
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: {prompt}")
    response = model.generate(prompt, max_length=100)
    print(f"🤖 Response: {response}")

print("\n✅ Generation test complete!")

# Save and reload LoRA adapters
"""
## 💾 Save Your Learned Knowledge!

LoRA adapters are like learning modules that can be saved and reused. Think of them
as skill cards that you can collect and apply to different models or tasks!

**What we're doing:**
- **Saving**: Extract just the LoRA weights (much smaller than full model)
- **Reloading**: Apply the learned adaptations to a fresh model
- **Verification**: Test that the reloaded model has the same behavior

**Why this matters:** LoRA adapters are:
- **Portable**: Can be shared and applied to different models
- **Efficient**: Much smaller than full model checkpoints
- **Flexible**: Can be combined or stacked for different tasks

This is like having a library of learned skills that you can mix and match!
"""

def save_reload_lora(
    model: Model,
    lora_path: str = "./lora.weights.lora.h5",
    preset: str = "gemma3_instruct_1b",
):
    """
    💾 Save and reload LoRA adapters for future use!

    This function demonstrates how to save your learned LoRA weights and then
    apply them to a fresh model. It's like saving a recipe and then using it
    to cook the same dish again!

    **The workflow:**
    1. Save LoRA weights from your trained model
    2. Load a fresh model from the preset
    3. Apply your saved LoRA weights
    4. Verify that both models produce the same output

    **Why this is powerful:** You can now reuse your fine-tuned knowledge
    without retraining, or share it with others!
    """
    print(f"💾 Saving LoRA adapters to {lora_path}")

    # Save LoRA adapters.
    model.backbone.save_lora_weights(lora_path)

    print("✅ LoRA adapters saved successfully!")

    # Reload the model and apply the saved LoRA weights.
    print("🔄 Reloading model and applying LoRA weights...")

    fresh_model = keras_hub.models.Gemma3CausalLM.from_preset(preset)
    fresh_model.backbone.load_lora_weights(lora_path)

    print("✅ LoRA weights loaded successfully!")

    # Test that the reloaded model produces the same output.
    test_prompt = "What is the capital of France?"
    original_response = model.generate(test_prompt, max_length=50)
    reloaded_response = fresh_model.generate(test_prompt, max_length=50)

    print(f"\n🧪 Testing LoRA reload:")
    print(f"📝 Prompt: {test_prompt}")
    print(f"🔴 Original: {original_response}")
    print(f"🟢 Reloaded: {reloaded_response}")
    print(f"✅ Responses match: {original_response == reloaded_response}")

    return fresh_model

# Execute this function to save and reload your LoRA adapters
reloaded_model = save_reload_lora(model, preset="gemma3_instruct_1b")

# Explore different model presets
"""
## 🎯 Explore Different Model Sizes!

Now that you've mastered the basics, let's explore different Gemma 3 model sizes!
Each preset offers different capabilities and trade-offs:

**Available Text Presets:**
- **gemma3_instruct_1b**: Perfect for learning and quick experiments
- **gemma3_instruct_4b_text**: Great for production tasks with more capacity
- **gemma3_instruct_8b_text**: High-quality results for demanding applications

**Why explore different sizes?**
- **Smaller models**: Train faster, use less memory, great for prototyping
- **Larger models**: Better quality, more capable, but require more resources
- **Finding the sweet spot**: Balance between performance and resource requirements

**Pro tip:** Start small and scale up as you need more capacity!
"""

# Execute this function to explore different model presets
def explore_model_presets():
    """
    🔍 Explore different Gemma 3 model presets to find the perfect fit!

    This function loads different model sizes so you can compare their characteristics
    and choose the best one for your specific use case.

    **What we'll discover:**
    - Parameter counts for different model sizes
    - Memory requirements and training characteristics
    - Quality vs. speed trade-offs

    **Why this matters:** Choosing the right model size is crucial for balancing
    performance, training time, and resource requirements!
    """
    presets = ["gemma3_instruct_1b", "gemma3_instruct_4b_text"]

    for preset in presets:
        print(f"\n🔍 Exploring {preset}...")
        try:
            model = keras_hub.models.Gemma3CausalLM.from_preset(preset)
            print(f"✅ Loaded successfully")
            print(f"📊 Parameters: {model.count_params():,}")
            print(f"🔧 Preprocessor: {model.preprocessor is not None}")
            print(f"📝 Preprocessor type: {type(model.preprocessor).__name__}")
        except Exception as e:
            print(f"❌ Failed to load: {e}")

    print("\n🎯 Choose the preset that best fits your needs!")

# Execute this function to explore different presets
explore_model_presets()

# Custom training with your own data
"""
## 📚 Train on Your Own Data!

Now it's time to apply what you've learned to your own projects! Here's how to
prepare and train on your own datasets:

**Data Format Requirements:**
- **prompts**: Your input text (questions, instructions, etc.)
- **responses**: Your desired outputs (answers, completions, etc.)
- **Format**: Must match the structure we used in our example

**Data Preparation Tips:**
- **Quality over quantity**: Better data beats more data
- **Consistent formatting**: Keep your prompts and responses consistent
- **Reasonable length**: Very long sequences can slow down training
- **Diverse examples**: Include different types of inputs and outputs

**Why this matters:** Your data is what makes the model useful for your specific
use case. Good data preparation is the foundation of successful fine-tuning!
"""

# Execute this function to create a custom dataset function
def create_custom_dataset(prompts, responses, batch_size=2):
    """
    🎨 Create a custom dataset from your own prompts and responses!

    This function takes your custom data and prepares it for training. You can use
    this to fine-tune the model on any text-based task you want!

    **Input format:**
    - prompts: List of input texts
    - responses: List of corresponding outputs
    - batch_size: How many examples to process together

    **Example usage:**
    ```python
    my_prompts = ["Your question 1", "Your question 2"]
    my_responses = ["Your answer 1", "Your answer 2"]
    custom_ds = create_custom_dataset(my_prompts, my_responses)
    ```

    **Why this is powerful:** You can now fine-tune the model on any domain,
    language, or task that matters to you!
    """
    features = {
        "prompts": prompts,
        "responses": responses,
    }

    # Create dataset using TensorFlow dataset API (compatible with all Keras 3 backends)
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.shuffle(len(prompts)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"✅ Custom dataset created with {len(prompts)} samples")
    print(f"📦 Batch size: {batch_size}")

    return ds

# Example: Create a custom dataset
example_prompts = [
    "What is Python?",
    "How do you write a function?",
    "Explain object-oriented programming",
]

example_responses = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "To write a function, use the 'def' keyword followed by the function name and parameters.",
    "Object-oriented programming is a paradigm that organizes code into objects with data and methods.",
]

# Execute this function to create your custom dataset
custom_ds = create_custom_dataset(example_prompts, example_responses, batch_size=1)

# Cell 13: Advanced distributed training options
"""
## 🚀 Advanced Distributed Training Options

Now that you're comfortable with the basics, let's explore some advanced options
for even more powerful distributed training!

**Advanced Distribution Strategies:**
- **Model Parallelism**: Split large models across devices when they don't fit
- **Multi-Worker Training**: Scale across multiple machines
- **Custom Device Meshes**: Fine-tune how work is distributed

**Performance Optimization:**
- **Batch Size Tuning**: Find the optimal batch size for your hardware
- **Gradient Accumulation**: Simulate larger batch sizes with less memory
- **Mixed Precision Tuning**: Optimize precision for your specific hardware

**Why explore these options?** They can unlock even more performance and enable
training of larger models that wouldn't fit on single devices!
"""

# Execute this function to explore advanced distribution options
def explore_advanced_distribution():
    """
    🚀 Explore how distribution strategies specifically benefit Gemma 3 fine-tuning!

    This function demonstrates how different distribution approaches can optimize
    your Gemma 3 fine-tuning workflow for maximum performance and efficiency.

    **What we'll explore:**
    - How distribution affects Gemma 3 training speed and memory usage
    - Performance differences when fine-tuning with LoRA across devices
    - Memory optimization strategies for large language models
    - Real-world scaling considerations for Gemma 3 fine-tuning

    **Why this matters:** Understanding distribution's impact on Gemma 3 training
    helps you choose the right strategy for your specific use case and hardware!
    """
    print("🚀 Gemma 3 Distributed Fine-tuning Optimization")
    print("=" * 60)

    # Check current distribution and its impact on Gemma 3 training
    print("📊 Current Distribution Impact on Gemma 3:")
    print(f"✅ DataParallel distribution is active")

    # Get device information relevant to Gemma 3 training
    try:
        devices = jax.devices()
        print(f"🚀 Available JAX devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")

        # Explain how this affects Gemma 3 training
        if len(devices) > 1:
            print(f"🎯 With {len(devices)} devices, your Gemma 3 training will:")
            print(f"   - Process {len(devices)}x more data per step")
            print(f"   - Reduce training time by ~{len(devices)}x")
            print(f"   - Enable larger effective batch sizes")
        else:
            print(f"ℹ️  Single device setup - consider multi-GPU for faster Gemma 3 training")

    except Exception as e:
        print(f"ℹ️  Device info: {e}")

    # Demonstrate distribution strategies specifically for Gemma 3
    print("\n🔄 Distribution Strategies for Gemma 3 Fine-tuning:")

    # Focus on strategies that matter for language model fine-tuning
    gemma3_strategies = {
        "DataParallel": {
            "class": keras.distribution.DataParallel,
            "best_for": "Most Gemma 3 fine-tuning scenarios",
            "benefits": "Faster training, larger batch sizes, easy scaling"
        },
        "ModelParallel": {
            "class": keras.distribution.ModelParallel,
            "best_for": "Very large Gemma 3 models (8B+ parameters)",
            "benefits": "Handle models too large for single devices"
        }
    }

    for name, info in gemma3_strategies.items():
        try:
            strategy = info["class"]()
            print(f"✅ {name}: Successfully created")
            print(f"   🎯 Best for: {info['best_for']}")
            print(f"   💡 Benefits: {info['benefits']}")

            # Test if we can set this distribution
            try:
                keras.distribution.set_distribution(strategy)
                print(f"   ✅ {name}: Successfully activated")

                # Switch back to DataParallel for our main workflow
                if name != "DataParallel":
                    keras.distribution.set_distribution(keras.distribution.DataParallel())
                    print(f"   🔄 Switched back to DataParallel")

            except Exception as e:
                print(f"   ❌ {name}: Failed to activate - {e}")

        except Exception as e:
            print(f"❌ {name}: Failed to create - {e}")

    # Gemma 3 specific performance insights
    print("\n📊 Gemma 3 Training Performance Insights:")
    print("Understanding how distribution affects your fine-tuning...")

    try:
        # Show how distribution affects batch size and memory
        print("\n💾 Memory and Batch Size Optimization:")
        print("   - Single device: Limited by device memory")
        print("   - DataParallel: Can use larger effective batch sizes")
        print("   - ModelParallel: Can handle larger models")

        # Explain LoRA benefits with distribution
        print("\n💎 LoRA + Distribution Synergy:")
        print("   - LoRA reduces memory per device")
        print("   - Distribution multiplies this benefit across devices")
        print("   - Result: Much faster fine-tuning with same memory")

        # Practical recommendations
        print("\n🎯 Practical Recommendations for Gemma 3:")
        if len(devices) == 1:
            print("   - Use LoRA to fit larger models in memory")
            print("   - Consider gradient accumulation for larger effective batches")
            print("   - Monitor memory usage during training")
        else:
            print("   - Leverage multiple devices for faster training")
            print("   - Scale batch size proportionally to device count")
            print("   - Use LoRA for efficient parameter updates")

    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")

    print("\n🎯 Distribution strategies optimized for Gemma 3 fine-tuning!")
    print("💡 Key takeaways:")
    print("   - DataParallel: Best for most Gemma 3 fine-tuning scenarios")
    print("   - LoRA + Distribution: Powerful combination for efficiency")
    print("   - Multiple devices: Dramatically faster training times")
    print("   - Memory optimization: Essential for large language models")

# Execute this function to explore advanced options
explore_advanced_distribution()

# Monitor and optimize performance
"""
## 📊 Monitor and Optimize Your Training!

Great training isn't just about running the code - it's about understanding what's
happening and making it better! Let's explore how to monitor and optimize your
distributed training performance.

**What to Monitor:**
- **Device utilization**: Are all your devices working efficiently?
- **Memory usage**: Are you using your available memory effectively?
- **Training speed**: How fast are you processing data?
- **Loss curves**: Is your model learning effectively?

**Optimization Strategies:**
- **Batch size tuning**: Find the sweet spot for your hardware
- **Learning rate adjustment**: Optimize the speed of learning
- **Mixed precision**: Balance speed and accuracy
- **Data pipeline optimization**: Keep your data flowing efficiently

**Why this matters:** Monitoring and optimization can double or triple your training
speed, making the difference between hours and days of training time!
"""

# Execute this function to set up performance monitoring
def setup_performance_monitoring():
    """
    📊 Set up comprehensive performance monitoring for your distributed training!

    This function helps you track the performance of your training and identify
    opportunities for optimization.

    **What we'll monitor:**
    - Training metrics and loss curves
    - Device utilization and memory usage
    - Data pipeline performance
    - Overall training efficiency

    **Why this matters:** Good monitoring helps you:
    - Identify bottlenecks and performance issues
    - Optimize your training configuration
    - Ensure you're getting the most from your hardware
    - Debug problems quickly and effectively
    """
    print("📊 Setting up Performance Monitoring")
    print("=" * 60)

    # Create comprehensive callbacks for monitoring
    callbacks = [
        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/ckpt-{epoch:02d}-{loss:.4f}.keras",
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
        ),

        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True,
        ),

        # Learning rate scheduling
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),

        # TensorBoard logging (if available)
        keras.callbacks.TensorBoard(
            log_dir="./logs",
            histogram_freq=1,
        ),
    ]

    print("✅ Performance monitoring callbacks created:")
    for i, callback in enumerate(callbacks, 1):
        print(f"  {i}. {type(callback).__name__}")

    print("\n🎯 These callbacks will help you monitor and optimize your training!")

    return callbacks

# Execute this function to set up monitoring
monitoring_callbacks = setup_performance_monitoring()

def generate_response(model, prompt, max_length=100):
    """
    🎯 Generate a response from a Gemma 3 model for a given prompt.

    This function handles the text generation process, providing a clean
    interface for testing model responses before and after fine-tuning.

    **Parameters:**
    - model: The Gemma 3 model to use for generation
    - prompt: The input text prompt
    - max_length: Maximum length of the generated response

    **Returns:** Generated text response from the model
    """
    try:
        # Generate response using the model
        response = model.generate(prompt, max_length=max_length)
        return response
    except Exception as e:
        return f"Error generating response: {e}"

def reload_lora_weights(model, lora_path):
    """
    💾 Reload LoRA weights from a saved file into a model.

    This function loads previously saved LoRA adapters and applies them
    to a model, enabling you to reuse fine-tuned weights.

    **Parameters:**
    - model: The model to load LoRA weights into
    - lora_path: Path to the saved LoRA weights file

    **Returns:** Model with LoRA weights loaded
    """
    try:
        # Load LoRA weights from the specified path
        model.load_weights(lora_path)
        print(f"✅ LoRA weights loaded from {lora_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load LoRA weights: {e}")
        return model

# Model Testing and Comparison
"""
## 🧪 Model Testing and Performance Comparison

Now let's test our models to see the real impact of fine-tuning! This section
demonstrates how to evaluate your models before and after training to measure
the effectiveness of your distributed fine-tuning efforts.

**What we'll accomplish:**
1. ✅ Test the original model's baseline performance
2. ✅ Test the fine-tuned model's improved capabilities
3. ✅ Compare responses side-by-side for analysis
4. ✅ Quantify improvements in response quality
5. ✅ Identify areas where fine-tuning had the most impact

**Why this matters:** Testing and comparison prove that your fine-tuning
was successful and help you understand what improvements were achieved!
"""

# Execute this function to run the complete training workflow
def test_model_before_finetuning():
    """
    🧪 Test the original Gemma 3 model before any fine-tuning!

    This cell demonstrates the baseline performance of your Gemma 3 model
    on various tasks before any training modifications.

    **What we'll test:**
    - General knowledge questions
    - Creative writing tasks
    - Technical explanations
    - Response quality and style

    **Why this matters:** Establishing a baseline helps you measure
    the effectiveness of your fine-tuning efforts!
    """
    print("🧪 Testing Original Gemma 3 Model (Before Fine-tuning)")
    print("=" * 60)

    # Load the base model without LoRA
    print("📥 Loading base Gemma 3 model...")
    base_model = create_model(TEXT_PRESET)

    # Test prompts covering different capabilities
    test_prompts = [
        "What is the capital of France?",
        "Write a short poem about artificial intelligence.",
        "Explain how neural networks work in simple terms.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]

    print(f"\n🎯 Testing {len(test_prompts)} different prompts...")
    print("📝 This shows the model's baseline capabilities:")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. 🎯 Prompt: {prompt}")
        print("   🤖 Response:")

        try:
            response = generate_response(base_model, prompt, max_length=150)
            # Clean up response for display
            clean_response = response.replace(prompt, "").strip()
            if clean_response:
                print(f"   {clean_response}")
            else:
                print("   (No additional response generated)")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n📊 Baseline Assessment Complete!")
    print("💡 This is your model's performance BEFORE fine-tuning.")
    print("🎯 Use these responses as a reference to measure improvements!")

    return base_model

def test_model_after_finetuning():
    """
    🚀 Test the fine-tuned Gemma 3 model to see the improvements!

    This cell demonstrates how your fine-tuning has changed the model's
    behavior and performance on the same tasks.

    **What we'll compare:**
    - Response quality improvements
    - Style and tone changes
    - Task-specific enhancements
    - Overall performance gains

    **Why this matters:** Seeing the before/after difference proves
    that your distributed fine-tuning was successful!
    """
    print("🚀 Testing Fine-tuned Gemma 3 Model (After Fine-tuning)")
    print("=" * 60)

    # Load the fine-tuned model (with LoRA weights)
    print("📥 Loading fine-tuned Gemma 3 model...")
    try:
        # Try to load existing LoRA weights
        fine_tuned_model = create_model(TEXT_PRESET)
        fine_tuned_model = enable_lora(fine_tuned_model, rank=4)

        # Check if we have saved weights
        if os.path.exists("./lora.weights.lora.h5"):
            print("💾 Found existing LoRA weights, loading them...")
            fine_tuned_model = reload_lora_weights(fine_tuned_model, "./lora.weights.lora.h5")
            print("✅ LoRA weights loaded successfully!")
        else:
            print("⚠️  No existing LoRA weights found.")
            print("💡 Run the training cell first to create fine-tuned weights!")
            return None

    except Exception as e:
        print(f"❌ Failed to load fine-tuned model: {e}")
        return None

    # Test the same prompts for fair comparison
    test_prompts = [
        "What is the capital of France?",
        "Write a short poem about artificial intelligence.",
        "Explain how neural networks work in simple terms.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis."
    ]

    print(f"\n🎯 Testing {len(test_prompts)} prompts on fine-tuned model...")
    print("📝 Compare these responses with the baseline:")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. 🎯 Prompt: {prompt}")
        print("   🤖 Fine-tuned Response:")

        try:
            response = generate_response(fine_tuned_model, prompt, max_length=150)
            # Clean up response for display
            clean_response = response.replace(prompt, "").strip()
            if clean_response:
                print(f"   {clean_response}")
            else:
                print("   (No additional response generated)")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n📊 Fine-tuned Assessment Complete!")
    print("💡 This is your model's performance AFTER fine-tuning.")
    print("🎯 Compare with baseline to see improvements!")

    return fine_tuned_model

def compare_before_after_performance():
    """
    📊 Direct comparison: Before vs After fine-tuning performance!

    This cell provides a side-by-side analysis of how your fine-tuning
    has improved the model's capabilities.

    **What we'll analyze:**
    - Response length and detail
    - Accuracy and relevance
    - Style and tone consistency
    - Task-specific improvements

    **Why this matters:** Quantifying improvements helps validate
    your fine-tuning approach and identify areas for further optimization!
    """
    print("📊 Before vs After Fine-tuning Performance Comparison")
    print("=" * 60)

    print("🔍 Loading both models for comparison...")

    # Load base model
    print("📥 Loading base model...")
    base_model = create_model(TEXT_PRESET)

    # Load fine-tuned model
    print("📥 Loading fine-tuned model...")
    fine_tuned_model = create_model(TEXT_PRESET)
    fine_tuned_model = enable_lora(fine_tuned_model, rank=4)

    if os.path.exists("./lora.weights.lora.h5"):
        fine_tuned_model = reload_lora_weights(fine_tuned_model, "./lora.weights.lora.h5")
        print("✅ Fine-tuned model loaded with LoRA weights!")
    else:
        print("❌ No fine-tuned weights found. Run training first!")
        return

    # Focus on a few key comparison prompts
    comparison_prompts = [
        "Explain machine learning in one sentence.",
        "What makes a good teacher?",
        "Describe the future of AI technology."
    ]

    print(f"\n🎯 Comparing responses on {len(comparison_prompts)} key prompts...")

    for i, prompt in enumerate(comparison_prompts, 1):
        print(f"\n{i}. 🎯 Prompt: {prompt}")
        print("   " + "="*50)

        # Get base model response
        print("🔴 BASE MODEL (Before Fine-tuning):")
        try:
            base_response = generate_response(base_model, prompt, max_length=100)
            base_clean = base_response.replace(prompt, "").strip()
            print(f"   {base_clean}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

        # Get fine-tuned model response
        print("🟢 FINE-TUNED MODEL (After Fine-tuning):")
        try:
            ft_response = generate_response(fine_tuned_model, prompt, max_length=100)
            ft_clean = ft_response.replace(prompt, "").strip()
            print(f"   {ft_clean}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

        # Analyze differences
        print("📊 ANALYSIS:")
        if base_clean != ft_clean:
            print("   ✅ Responses differ - fine-tuning has an effect!")

            # Simple metrics
            base_words = len(base_clean.split())
            ft_words = len(ft_clean.split())

            print(f"   📏 Base response: {base_words} words")
            print(f"   📏 Fine-tuned response: {ft_words} words")

            if ft_words > base_words:
                print("   📈 Fine-tuned model provides more detailed responses!")
            elif ft_words < base_words:
                print("   📉 Fine-tuned model provides more concise responses!")
            else:
                print("   📊 Response length is similar")
        else:
            print("   ℹ️  Responses are identical - may need more training data")

        print("   " + "-"*50)

    print("\n🎯 Comparison Complete!")
    print("💡 Key insights:")
    print("   - Look for response quality improvements")
    print("   - Notice style and tone changes")
    print("   - Identify areas where fine-tuning helped most")
    print("   - Use this analysis to guide future training iterations!")

# Execute these functions to test and compare model performance
# test_model_before_finetuning()
# test_model_after_finetuning()
# compare_before_after_performance()

"""
## 🌟 Congratulations! You've Completed the Journey!

🎉 **Amazing work!** You've successfully navigated the exciting world of distributed
fine-tuning with Keras Hub and Gemma 3 models. Let's take a moment to celebrate what
you've accomplished and look ahead to what's next!

### **🏆 What You've Mastered:**

🚀 **Distributed Training Fundamentals**
- Understanding how to scale training across multiple devices
- Mastering the Keras 3 Distribution API
- Learning to harness the power of data parallelism

💎 **Parameter-Efficient Fine-tuning**
- Implementing LoRA for memory-efficient training
- Saving and reloading learned adaptations
- Balancing model capacity with computational efficiency

🎯 **Model Management with Keras Hub**
- Loading pre-trained models from presets
- Automatic preprocessor and tokenizer integration
- Seamless fine-tuning workflows

### **🚀 What's Next on Your AI Journey:**

**Immediate Next Steps:**
- Experiment with different LoRA ranks and learning rates
- Try fine-tuning on your own datasets
- Explore different model presets and sizes

**Advanced Explorations:**
- Implement custom training loops with distributed training
- Experiment with model parallelism for larger models
- Explore multi-worker distributed training across machines

**Real-World Applications:**
- Fine-tune models for your specific domain or task
- Build production pipelines for continuous model improvement
- Contribute to the open-source AI community

### **💡 Key Insights You've Gained:**

1. **Distributed training isn't just about speed** - it's about making previously
   impossible tasks possible and opening new frontiers in AI development.

2. **Keras 3 represents the future** - a unified, backend-agnostic approach that
   gives you the flexibility to choose the best tools for your specific needs.

3. **Parameter-efficient fine-tuning** - LoRA and similar techniques democratize
   AI by making powerful models accessible to more developers and researchers.

4. **The power of integration** - Keras Hub's seamless integration of models,
   preprocessors, and tokenizers eliminates the complexity that often
   prevents people from experimenting with cutting-edge AI.

### **🎯 Remember:**

**The journey never truly ends** - AI is evolving rapidly, and there are always
new techniques, models, and possibilities to explore. What you've learned here
is just the beginning of your AI adventure!

**Keep experimenting, keep learning, and keep building amazing things!**

The world of AI is waiting for your contributions, and you now have the tools
and knowledge to make a real difference. Whether you're building the next
breakthrough application, contributing to research, or teaching others, you're
part of the AI revolution that's shaping our future.

**Thank you for joining this incredible journey!** 🚀✨

---

## 🔧 Quick Reference

**Key Functions:**
- `set_data_parallel_distribution()`: Set up distributed training
- `create_model(preset)`: Load a Gemma 3 model with preprocessor
- `enable_lora(model, rank)`: Enable LoRA for efficient fine-tuning
- `build_text_only_dataset()`: Create training datasets
- `compile_model()`: Configure training parameters
- `save_reload_lora()`: Save and reload LoRA adapters
- `test_model_before_finetuning()`: Test baseline model performance
- `test_model_after_finetuning()`: Test fine-tuned model performance
- `compare_before_after_performance()`: Side-by-side performance comparison

**Distribution Strategies:**
- `keras.distribution.DataParallel`: Data parallelism across devices
- `keras.distribution.ModelParallel`: Model parallelism for large models
- `keras.distribution.MultiWorkerDataParallel`: Multi-machine scaling

**Best Practices:**
- Always enable mixed precision for better performance
- Use LoRA for memory-efficient fine-tuning
- Start with small models and datasets for learning
- Monitor device utilization during training
- Save LoRA adapters for future reuse

---
**The power is in your hands now!** 🌟✨
"""

