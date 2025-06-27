#!/usr/bin/env python3
"""
Test Model Loading - Diagnose MusicGen loading issues
"""

import os
import sys
import torch
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

print("🔍 MusicGen Model Loading Test")
print("="*50)

# 1. Check environment
print("\n1. Environment Check:")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__ if 'torch' in sys.modules else 'Not installed'}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current directory: {os.getcwd()}")
print(f"Home directory: {os.path.expanduser('~')}")

# 2. Check transformers
print("\n2. Checking transformers library:")
try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
    
    # Check if MusicGen is available
    from transformers import MusicgenForConditionalGeneration
    print("✓ MusicgenForConditionalGeneration available")
    
    from transformers import AutoProcessor
    print("✓ AutoProcessor available")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing transformers...")
    os.system(f"{sys.executable} -m pip install --upgrade transformers")
    sys.exit(1)

# 3. Test downloading config only
print("\n3. Testing config download:")
try:
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(
        "facebook/musicgen-small",
        cache_dir="./test_cache"
    )
    print("✓ Config downloaded successfully")
    print(f"  Model type: {config.model_type}")
    
except Exception as e:
    print(f"❌ Config download failed: {e}")

# 4. Test processor loading
print("\n4. Testing processor loading:")
try:
    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained(
        "facebook/musicgen-small",
        cache_dir="./test_cache"
    )
    print("✓ Processor loaded successfully")
    
    # Test tokenization
    inputs = processor(text=["test prompt"], return_tensors="pt")
    print(f"✓ Tokenization works: {inputs.input_ids.shape}")
    
except Exception as e:
    print(f"❌ Processor loading failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Test minimal model loading
print("\n5. Testing minimal model loading:")
print("   (This downloads the full model, ~1.5GB)")

try:
    # Set environment for better compatibility
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    from transformers import MusicgenForConditionalGeneration
    
    # Try simplest loading approach
    print("   Attempting to load model...")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        cache_dir="./test_cache",
        torch_dtype=torch.float32  # Use float32 for compatibility
    )
    
    print("✓ Model loaded successfully!")
    
    # Check model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count/1e6:.1f}M")
    print(f"  Device: {next(model.parameters()).device}")
    
    # Test generation capability
    print("\n6. Testing generation capability:")
    inputs = processor(text=["test"], return_tensors="pt")
    
    with torch.no_grad():
        # Generate just 1 token to test
        output = model.generate(**inputs, max_new_tokens=1)
        print(f"✓ Generation test passed: output shape {output.shape}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("MusicGen is working correctly on your system.")
    
except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 Troubleshooting suggestions:")
    print("1. Check internet connection")
    print("2. Clear cache: rm -rf test_cache")
    print("3. Update transformers: pip install --upgrade transformers")
    print("4. Check disk space (need ~2GB free)")
    print("5. Try with VPN if in restricted region")

print("\n" + "="*50)