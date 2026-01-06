#!/usr/bin/env python3
"""
V5.8 Training Script - CURATED (V5.2 Quality + V5.7 Coverage)

Key Improvements from V5.7:
1. 100% curated examples (V5.2's strict quality filters applied)
2. 1,296 examples (81.4% of V5.7 passed quality checks)
3. 8 categories (same as V5.7)
4. All examples meet V5.2's proven quality standards

Quality Filter Results:
- Background Tasks:   100% pass rate (225/225)
- Error Handling:     100% pass rate (225/225)
- WebSockets:         100% pass rate (225/225)
- Async SQLAlchemy:   95.1% pass rate (214/225)
- JWT Authentication: 68.0% pass rate (153/225)
- Query Optimization: 64.0% pass rate (144/225)
- CRUD:               62.7% pass rate (141/225)
- Pagination:         61.3% pass rate (138/225)

Dataset: 1,296 examples (100% curated, V5.2-level quality)
Expected: 88-92% overall accuracy across 8 categories
"""

import os
import json
import torch
import numpy as np
import random
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# CRITICAL: SET RANDOM SEEDS FIRST
# =============================================================================
SEED = 42

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ All random seeds set to {seed}")
    return seed

# SET SEED BEFORE ANY OTHER OPERATIONS
print()
print("="*80)
print("🔐 SETTING RANDOM SEEDS FOR REPRODUCIBILITY")
print("="*80)
set_seed(SEED)
print("="*80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "/home/ec2-user/1BFINE/data/fastapi_1b_v5_8_curated.jsonl"
OUTPUT_DIR = "/home/ec2-user/1BFINE/models/llama-1b-fastapi-v5-8"

# Training Parameters (optimized for 1,296 examples)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 1536

# Checkpoints (adjusted for 1,296 examples)
SAVE_STEPS = 80   # Save every 80 steps
EVAL_STEPS = 80
LOGGING_STEPS = 10

# LoRA Configuration (proven from V5.2)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =============================================================================
# HEADER
# =============================================================================
print("="*80)
print("🚀 V5.8 TRAINING - CURATED (V5.2 Quality + V5.7 Coverage)")
print("="*80)
print()
print("📋 V5.8 Strategy (Learning from V5.6 & V5.7):")
print("   ✅ 1,296 examples (100% passed V5.2's strict filters)")
print("   ✅ 8 categories (1B model sweet spot)")
print("   ✅ V5.2-level quality standards")
print("   ✅ No template artifacts or low-quality examples")
print()
print("🔍 Quality Filter Results (V5.7 → V5.8):")
print("   ✅ Started: 1,800 examples")
print("   ✅ Applied: V5.2's strict quality criteria")
print("   ✅ Passed:  1,465 examples (81.4%)")
print("   ✅ Final:   1,296 examples (balanced)")
print()
print("📊 Pass Rates by Category:")
print("   ✅ Background Tasks:   100% (225/225)")
print("   ✅ Error Handling:     100% (225/225)")
print("   ✅ WebSockets:         100% (225/225)")
print("   ✅ Async SQLAlchemy:    95% (214/225)")
print("   ⚠️  JWT Auth:            68% (153/225)")
print("   ⚠️  Query Opt:           64% (144/225)")
print("   ⚠️  CRUD:                63% (141/225)")
print("   ⚠️  Pagination:          61% (138/225)")
print()
print("🎯 Target Categories (8):")
print("   1. Async SQLAlchemy         (180 examples)")
print("   2. Background Tasks         (180 examples)")
print("   3. Error Handling           (180 examples)")
print("   4. WebSockets               (180 examples)")
print("   5. JWT Authentication       (153 examples)")
print("   6. Query Optimization       (144 examples)")
print("   7. CRUD Operations          (141 examples)")
print("   8. Pagination               (138 examples)")
print()
print("📊 Training Configuration:")
print(f"   Model:                  {MODEL_NAME}")
print(f"   Dataset:                {DATASET_PATH}")
print(f"   Dataset Size:           1,296 examples")
print(f"   Batch Size:             {BATCH_SIZE}")
print(f"   Gradient Accumulation:  {GRADIENT_ACCUMULATION_STEPS}")
print(f"   Effective Batch:        {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Learning Rate:          {LEARNING_RATE}")
print(f"   Epochs:                 {NUM_EPOCHS}")
print(f"   Max Length:             {MAX_LENGTH}")
print(f"   LoRA R:                 {LORA_R}")
print(f"   LoRA Alpha:             {LORA_ALPHA}")
print(f"   Random Seed:            {SEED}")
print()
print("⏱️  Expected Training Time:")
print("   Steps per epoch:        ~81 steps")
print("   Total steps:            ~243 steps (3 epochs)")
print("   Time per step:          ~18 seconds")
print("   Total time:             ~1.2 hours")
print("   Cost:                   ~$3.38")
print()
print("🎯 Expected Results:")
print("   Overall accuracy:       88-92%")
print("   All categories:         85-95% each")
print("   Quality:                V5.2-level (100% curated)")
print()
print("="*80)
print()

# =============================================================================
# LOAD AND PREPARE DATASET
# =============================================================================
print("📂 Loading dataset...")
print(f"   Path: {DATASET_PATH}")

# Load dataset
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
print(f"   ✅ Loaded {len(dataset)} examples")

# Verify dataset
print("\n🔍 Dataset verification:")
sample = dataset[0]
print(f"   Sample keys: {list(sample.keys())}")
print(f"   Instruction: {sample.get('instruction', 'N/A')[:80]}...")
print(f"   Output length: {len(sample.get('output', ''))} chars")
print(f"   Category: {sample.get('category', 'N/A')}")

# Split dataset (90% train, 10% eval)
print("\n✂️  Splitting dataset...")
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = dataset['train']
eval_dataset = dataset['test']
print(f"   Train: {len(train_dataset)} examples")
print(f"   Eval:  {len(eval_dataset)} examples")

# =============================================================================
# PREPARE TOKENIZER AND FORMAT DATASET
# =============================================================================
print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("   ✅ Tokenizer loaded")

def format_instruction(example):
    """Format examples into instruction-response format"""
    instruction = example.get('instruction', '')
    output = example.get('output', '')
    
    # Use the proven format from V2/V5.2
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
    
    return {"text": text}

print("\n🔄 Formatting dataset...")
train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_instruction, remove_columns=eval_dataset.column_names)
print("   ✅ Dataset formatted")

# Tokenize
print("\n🎯 Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train dataset"
)

eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing eval dataset"
)
print("   ✅ Tokenization complete")

# =============================================================================
# LOAD AND PREPARE MODEL
# =============================================================================
print("\n🤖 Loading base model...")
print(f"   Model: {MODEL_NAME}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
print("   ✅ Base model loaded")

# Prepare model for training
print("\n🔧 Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("   ✅ Model prepared")

# Apply LoRA
print("\n🎛️  Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("   ✅ LoRA applied")

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📊 Model parameters:")
print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print(f"   Total:     {total_params:,}")

# =============================================================================
# TRAINING ARGUMENTS
# =============================================================================
print("\n⚙️  Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    
    # Optimization
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Logging and saving
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    save_total_limit=3,
    
    # Evaluation
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Performance
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    
    # Reproducibility
    seed=SEED,
    data_seed=SEED,
    
    # Reporting
    report_to="none",
    logging_dir=f"{OUTPUT_DIR}/logs",
    
    # Don't remove unused columns (we already handled this)
    remove_unused_columns=False,
)

print("   ✅ Training arguments configured")

# =============================================================================
# DATA COLLATOR
# =============================================================================
print("\n📦 Setting up data collator...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
print("   ✅ Data collator ready")

# =============================================================================
# TRAINER
# =============================================================================
print("\n👨‍🏫 Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
print("   ✅ Trainer initialized")

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print()
print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Training {len(train_dataset)} examples for {NUM_EPOCHS} epochs")
print(f"⏱️  Expected duration: ~1.2 hours")
print(f"💾 Checkpoints every {SAVE_STEPS} steps")
print(f"📈 Evaluation every {EVAL_STEPS} steps")
print()
print("="*80)
print()

# Save metadata
metadata = {
    "model_name": MODEL_NAME,
    "dataset_path": DATASET_PATH,
    "dataset_size": len(dataset),
    "train_size": len(train_dataset),
    "eval_size": len(eval_dataset),
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "max_length": MAX_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "seed": SEED,
    "start_time": datetime.now().isoformat(),
    "target_categories": [
        "Async SQLAlchemy",
        "Background Tasks",
        "Error Handling",
        "WebSockets",
        "JWT Authentication",
        "Query Optimization",
        "CRUD Operations",
        "Pagination"
    ],
    "total_categories": 8,
    "quality_approach": "V5.2 strict curation applied to V5.7",
    "quality_pass_rate": "81.4% (1465/1800 examples passed)",
    "improvements_from_v5_7": [
        "Applied V5.2's strict quality filters",
        "Removed 335 low-quality examples",
        "100% curated examples (V5.2-level quality)",
        "No template artifacts"
    ],
    "expected_training_time_hours": 1.2,
    "expected_accuracy": "88-92%",
    "version": "V5.8_curated"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/training_metadata_v5_8.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Train!
try:
    trainer.train()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print()
    print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"   ✅ Model saved to {OUTPUT_DIR}/final")
    
    # Update metadata
    metadata["end_time"] = datetime.now().isoformat()
    metadata["status"] = "completed"
    with open(f"{OUTPUT_DIR}/training_metadata_v5_8.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 V5.8 TRAINING SUCCESSFUL!")
    print("="*80)
    print()
    print("📊 Next Steps:")
    print("   1. Test V5.8 model on all 8 categories")
    print("   2. Compare with V5.2 (94.4%, 5 cats) and V5.6 (69.2%, 12 cats)")
    print("   3. Expected: 88-92% overall accuracy")
    print()
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ TRAINING FAILED!")
    print("="*80)
    print(f"\nError: {str(e)}")
    
    # Update metadata
    metadata["end_time"] = datetime.now().isoformat()
    metadata["status"] = "failed"
    metadata["error"] = str(e)
    with open(f"{OUTPUT_DIR}/training_metadata_v5_8.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    raise

