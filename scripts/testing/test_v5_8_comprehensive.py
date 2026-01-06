#!/usr/bin/env python3
"""
V5.8 Comprehensive Testing - 8 Categories

Tests V5.8 (curated, 100% quality) on all 8 target categories:
1. CRUD Operations
2. Async SQLAlchemy
3. JWT Authentication
4. Query Optimization
5. Pagination
6. WebSockets
7. Error Handling
8. Background Tasks

Expected: 88-92% overall accuracy
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# Model path
MODEL_PATH = "/home/ec2-user/1BFINE/models/llama-1b-fastapi-v5-8/final"

print("="*80)
print("🧪 V5.8 COMPREHENSIVE TESTING - 8 CATEGORIES")
print("="*80)
print()
print("   Model: V5.8 Curated (100% quality)")
print("   Categories: 8")
print("   Expected: 88-92% accuracy")
print()
print("="*80)
print()

# Load model
print("📦 Loading V5.8 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("   ✅ Model loaded")
print()

def generate_code(instruction):
    """Generate code from instruction"""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    return response

def check_patterns(code, patterns):
    """Check if code contains required patterns"""
    results = {}
    for name, pattern in patterns.items():
        # All patterns starting with r' are regex patterns
        if pattern.startswith("r") or "\\" in pattern or "|" in pattern or "(" in pattern:
            # It's a regex pattern
            results[name] = bool(re.search(pattern, code, re.IGNORECASE | re.DOTALL))
        else:
            # Simple string search (case-insensitive)
            results[name] = pattern.lower() in code.lower()
    return results

# =============================================================================
# TEST CASES - 8 CATEGORIES
# =============================================================================

test_cases = [
    # 1. CRUD Operations
    {
        "category": "CRUD",
        "instruction": "Create a FastAPI POST endpoint to create a new user with username and email using SQLAlchemy.",
        "required_patterns": {
            "db.add": r"db\.add\(",
            "db.commit": r"db\.commit\(\)",
            "Session": "session",
            "POST": "@app.post",
        },
        "anti_patterns": {
            "in_memory": r"=\s*\[\]|\.append\("
        }
    },
    {
        "category": "CRUD",
        "instruction": "Create a GET endpoint to fetch a single product by ID from the database with error handling.",
        "required_patterns": {
            "filter": r"\.filter\(",
            "HTTPException": "httpexception",
            "404": "404",
        }
    },
    {
        "category": "CRUD",
        "instruction": "Create a PUT endpoint to update a user's email in the database.",
        "required_patterns": {
            "db.commit": r"db\.commit\(\)",
            "filter": r"\.filter\(",
            "Session": "session",
        }
    },
    
    # 2. Async SQLAlchemy
    {
        "category": "Async SQLAlchemy",
        "instruction": "Create an async FastAPI endpoint that fetches all users using AsyncSession and SQLAlchemy.",
        "required_patterns": {
            "AsyncSession": "asyncsession",
            "async def": r"async\s+def",
            "await": r"\bawait\b",
        }
    },
    {
        "category": "Async SQLAlchemy",
        "instruction": "Implement async database operations with proper async context manager and error handling.",
        "required_patterns": {
            "AsyncSession": "asyncsession",
            "async with": r"async\s+with",
            "await": r"\bawait\b",
        }
    },
    {
        "category": "Async SQLAlchemy",
        "instruction": "Create async CRUD endpoint with SQLAlchemy AsyncSession for creating orders.",
        "required_patterns": {
            "AsyncSession": "asyncsession",
            "async def": r"async\s+def",
            "await": r"\bawait\b",
            "add": r"\.add\(",
        }
    },
    
    # 3. JWT Authentication
    {
        "category": "JWT Authentication",
        "instruction": "Create a FastAPI login endpoint that returns JWT tokens with proper security.",
        "required_patterns": {
            "jwt": r"\bjwt\b",
            "encode": r"\.encode\(",
            "token": "token",
        }
    },
    {
        "category": "JWT Authentication",
        "instruction": "Implement JWT token validation and create a protected endpoint.",
        "required_patterns": {
            "jwt": r"\bjwt\b",
            "decode": r"\.decode\(",
            "HTTPException": "httpexception",
        }
    },
    {
        "category": "JWT Authentication",
        "instruction": "Create a JWT authentication dependency for protecting routes.",
        "required_patterns": {
            "jwt": r"\bjwt\b",
            "Depends": "depends",
            "decode": r"\.decode\(",
        }
    },
    
    # 4. Query Optimization
    {
        "category": "Query Optimization",
        "instruction": "Create a FastAPI endpoint to fetch users with their posts using joinedload to avoid N+1 queries.",
        "required_patterns": {
            "joinedload": "joinedload",
            "options": r"\.options\(",
        }
    },
    {
        "category": "Query Optimization",
        "instruction": "Implement eager loading with selectinload for posts and comments relationship.",
        "required_patterns": {
            "selectinload": "selectinload",
            "options": r"\.options\(",
        }
    },
    {
        "category": "Query Optimization",
        "instruction": "Optimize database query to load user with related orders using eager loading.",
        "required_patterns": {
            "joinedload|selectinload": r"(joinedload|selectinload)",
            "options": r"\.options\(",
        }
    },
    
    # 5. Pagination
    {
        "category": "Pagination",
        "instruction": "Create a paginated endpoint to list users with skip and limit parameters.",
        "required_patterns": {
            "skip": r"skip:\s*int",
            "limit": r"limit:\s*int",
            "offset": r"\.offset\(",
            "limit_query": r"\.limit\(",
        }
    },
    {
        "category": "Pagination",
        "instruction": "Implement pagination for products listing with page number and page size.",
        "required_patterns": {
            "skip|page": r"(skip|page)",
            "limit|size": r"(limit|size)",
            "offset": r"\.offset\(",
        }
    },
    {
        "category": "Pagination",
        "instruction": "Create cursor-based pagination for posts using ID-based cursors.",
        "required_patterns": {
            "cursor": "cursor",
            "filter": r"\.filter\(",
            "limit": r"\.limit\(",
        }
    },
    
    # 6. WebSockets
    {
        "category": "WebSockets",
        "instruction": "Create a WebSocket endpoint for real-time chat with connection management.",
        "required_patterns": {
            "WebSocket": "websocket",
            "async def": r"async\s+def",
            "await": r"\bawait\b",
            "accept": r"\.accept\(",
        }
    },
    {
        "category": "WebSockets",
        "instruction": "Implement WebSocket connection manager for broadcasting messages to multiple clients.",
        "required_patterns": {
            "WebSocket": "websocket",
            "async": r"\basync\b",
            "await": r"\bawait\b",
        }
    },
    {
        "category": "WebSockets",
        "instruction": "Create WebSocket endpoint with disconnect handling and message routing.",
        "required_patterns": {
            "WebSocket": "websocket",
            "receive|send": r"(receive|send)",
            "await": r"\bawait\b",
        }
    },
    
    # 7. Error Handling
    {
        "category": "Error Handling",
        "instruction": "Create custom exception handlers for FastAPI with structured error responses.",
        "required_patterns": {
            "exception_handler": r"@app\.exception_handler",
            "status_code": "status_code",
        }
    },
    {
        "category": "Error Handling",
        "instruction": "Implement global exception handling with logging for FastAPI.",
        "required_patterns": {
            "exception_handler": r"exception_handler",
            "status_code": "status_code",
            "Exception": "exception",
        }
    },
    {
        "category": "Error Handling",
        "instruction": "Create validation error handler with detailed error messages.",
        "required_patterns": {
            "HTTPException|ValidationError": r"(httpexception|validationerror)",
            "status_code": "status_code",
        }
    },
    
    # 8. Background Tasks
    {
        "category": "Background Tasks",
        "instruction": "Create a FastAPI endpoint that sends email notifications in the background.",
        "required_patterns": {
            "BackgroundTasks": "backgroundtasks",
            "add_task": r"\.add_task\(",
            "def ": r"def\s+\w+",
        }
    },
    {
        "category": "Background Tasks",
        "instruction": "Implement background task for data processing after user upload.",
        "required_patterns": {
            "BackgroundTasks": "backgroundtasks",
            "add_task": r"\.add_task\(",
        }
    },
    {
        "category": "Background Tasks",
        "instruction": "Create endpoint with background task for sending notifications.",
        "required_patterns": {
            "BackgroundTasks": "backgroundtasks",
            "add_task": r"\.add_task\(",
            "async|def": r"(async\s+)?def",
        }
    },
]

# =============================================================================
# RUN TESTS
# =============================================================================

print("🧪 Running comprehensive tests...")
print()

results_by_category = {}
total_passed = 0
total_tests = len(test_cases)

for i, test in enumerate(test_cases, 1):
    category = test["category"]
    instruction = test["instruction"]
    required = test.get("required_patterns", {})
    anti = test.get("anti_patterns", {})
    
    print(f"Test {i}/{total_tests}: {category}")
    print(f"   Instruction: {instruction[:60]}...")
    
    # Generate code
    generated = generate_code(instruction)
    
    # DEBUG: Print first 200 chars of generated code
    if i == 1:  # Only for first test
        print(f"   DEBUG Generated (first 200 chars): {generated[:200]}")
    
    # Check required patterns
    required_results = check_patterns(generated, required)
    required_pass = all(required_results.values())
    
    # Check anti-patterns (should NOT be present)
    anti_pass = True
    if anti:
        anti_results = check_patterns(generated, anti)
        anti_pass = not any(anti_results.values())  # All should be False
    
    # Overall pass
    passed = required_pass and anti_pass
    
    if passed:
        print(f"   ✅ PASS")
        total_passed += 1
    else:
        print(f"   ❌ FAIL")
        if not required_pass:
            missing = [k for k, v in required_results.items() if not v]
            print(f"      Missing: {', '.join(missing)}")
        if not anti_pass:
            print(f"      Has anti-patterns (bad patterns found)")
    
    # Track by category
    if category not in results_by_category:
        results_by_category[category] = {"passed": 0, "total": 0}
    results_by_category[category]["total"] += 1
    if passed:
        results_by_category[category]["passed"] += 1
    
    print()

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("="*80)
print("📊 V5.8 TEST RESULTS SUMMARY")
print("="*80)
print()

print("By Category:")
print("─" * 70)
print(f"{'Category':<25} {'Passed':<10} {'Total':<10} {'Accuracy':<10}")
print("─" * 70)

for cat in sorted(results_by_category.keys()):
    stats = results_by_category[cat]
    passed = stats["passed"]
    total = stats["total"]
    accuracy = (passed / total * 100) if total > 0 else 0
    status = "✅" if accuracy >= 85 else "⚠️" if accuracy >= 70 else "❌"
    print(f"{status} {cat:<23} {passed:<10} {total:<10} {accuracy:>6.1f}%")

print("─" * 70)
overall_accuracy = (total_passed / total_tests * 100) if total_tests > 0 else 0
status = "✅" if overall_accuracy >= 88 else "⚠️" if overall_accuracy >= 80 else "❌"
print(f"{status} {'OVERALL':<23} {total_passed:<10} {total_tests:<10} {overall_accuracy:>6.1f}%")
print("─" * 70)
print()

# Comparison
print("📈 Comparison with Other Models:")
print("─" * 70)
print("   V5.2:  94.4% (5 categories) ✅")
print("   V5.6:  69.2% (12 categories) ❌")
print(f"   V5.8:  {overall_accuracy:.1f}% (8 categories) {'✅' if overall_accuracy >= 88 else '⚠️'}")
print("─" * 70)
print()

# Verdict
print("🎯 Verdict:")
if overall_accuracy >= 90:
    print("   ⭐ EXCELLENT! Exceeds expectations!")
    print("   V5.8 is production-ready!")
elif overall_accuracy >= 88:
    print("   ✅ SUCCESS! Meets expectations (88-92%)!")
    print("   V5.8 is production-ready!")
elif overall_accuracy >= 80:
    print("   ⚠️  GOOD, but below target (88-92%)")
    print("   V5.8 is usable, but could be improved")
else:
    print("   ❌ BELOW EXPECTATIONS")
    print("   Further improvements needed")
print()

print("="*80)
print("✅ TESTING COMPLETE")
print("="*80)

