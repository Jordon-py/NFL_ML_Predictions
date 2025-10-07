#!/usr/bin/env python3
"""
Test script to validate enhanced pipeline structure and components.
Run this to verify the pipeline is properly installed.
"""

import ast
import sys
from pathlib import Path

def validate_file_structure():
    """Verify all required files exist."""
    print("="*60)
    print("VALIDATING FILE STRUCTURE")
    print("="*60)
    
    required_files = [
        "backend/enhanced_pipeline.py",
        "docs/report.md",
        "docs/clean_data_example.ipynb",
        "backend/PIPELINE_README.md"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = Path(file)
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"✗ {file} NOT FOUND")
            all_exist = False
    
    return all_exist

def validate_pipeline_code():
    """Validate enhanced_pipeline.py structure."""
    print("\n" + "="*60)
    print("VALIDATING PIPELINE CODE STRUCTURE")
    print("="*60)
    
    with open('backend/enhanced_pipeline.py', 'r') as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    
    # Extract classes and functions
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    print(f"✓ Syntax validation passed")
    print(f"✓ Found {len(classes)} classes, {len(functions)} functions")
    
    # Validate required classes
    required_classes = [
        'RecoverySolution',
        'PipelineError',
        'DatasetValidator',
        'DatasetMerger',
        'ModelEvaluator'
    ]
    
    print("\nRequired classes:")
    all_classes_found = True
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ {cls}")
        else:
            print(f"  ✗ {cls} NOT FOUND")
            all_classes_found = False
    
    # Validate required functions
    required_functions = [
        'run_enhanced_pipeline',
        'main'
    ]
    
    print("\nRequired functions:")
    all_functions_found = True
    for func in required_functions:
        if func in functions:
            print(f"  ✓ {func}")
        else:
            print(f"  ✗ {func} NOT FOUND")
            all_functions_found = False
    
    return all_classes_found and all_functions_found

def validate_error_recovery():
    """Validate error recovery system."""
    print("\n" + "="*60)
    print("VALIDATING ERROR RECOVERY SYSTEM")
    print("="*60)
    
    with open('backend/enhanced_pipeline.py', 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find error types handled
    error_types = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PipelineError':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '_generate_solutions':
                    for child in ast.walk(item):
                        if isinstance(child, ast.Compare):
                            for op in child.ops:
                                if isinstance(op, ast.Eq):
                                    if hasattr(child.left, 'attr') and child.left.attr == 'error_type':
                                        if isinstance(child.comparators[0], ast.Constant):
                                            error_types.append(child.comparators[0].value)
    
    expected_errors = ['DATATYPE_MISMATCH', 'JOIN_ERROR', 'MISSING_FEATURES', 'FEATURE_LEAKAGE']
    
    print(f"✓ Found {len(set(error_types))} distinct error handlers")
    print("\nError types covered:")
    
    all_covered = True
    for et in expected_errors:
        if et in error_types:
            print(f"  ✓ {et}")
        else:
            print(f"  ⚠ {et} (handled by generic handler)")
    
    return len(set(error_types)) == len(expected_errors)

def validate_dataset_validator():
    """Validate DatasetValidator class."""
    print("\n" + "="*60)
    print("VALIDATING DATASET VALIDATOR")
    print("="*60)
    
    with open('backend/enhanced_pipeline.py', 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find DatasetValidator class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'DatasetValidator':
            methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
            
            expected_validations = [
                'validate_schema',
                'validate_datatypes',
                'validate_team_codes',
                'validate_temporal_order',
                'validate_no_leakage'
            ]
            
            print(f"✓ Found DatasetValidator with {len(methods)} methods")
            print("\nValidation methods:")
            
            all_found = True
            for val in expected_validations:
                if val in methods:
                    print(f"  ✓ {val}")
                else:
                    print(f"  ✗ {val} NOT FOUND")
                    all_found = False
            
            return all_found
    
    print("✗ DatasetValidator class not found")
    return False

def validate_documentation():
    """Validate documentation files."""
    print("\n" + "="*60)
    print("VALIDATING DOCUMENTATION")
    print("="*60)
    
    docs = {
        "docs/report.md": ["Usage", "Error Recovery", "Architecture"],
        "backend/PIPELINE_README.md": ["Quick Start", "Error Recovery Examples", "Troubleshooting"]
    }
    
    all_valid = True
    for doc_file, required_sections in docs.items():
        if Path(doc_file).exists():
            with open(doc_file, 'r') as f:
                content = f.read()
            
            print(f"\n{doc_file}:")
            for section in required_sections:
                if section in content:
                    print(f"  ✓ Contains '{section}' section")
                else:
                    print(f"  ✗ Missing '{section}' section")
                    all_valid = False
        else:
            print(f"\n✗ {doc_file} not found")
            all_valid = False
    
    return all_valid

def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("ENHANCED PIPELINE VALIDATION TEST")
    print("="*60)
    print()
    
    results = {
        "File Structure": validate_file_structure(),
        "Pipeline Code": validate_pipeline_code(),
        "Error Recovery": validate_error_recovery(),
        "Dataset Validator": validate_dataset_validator(),
        "Documentation": validate_documentation()
    }
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("="*60)
        print("\nThe enhanced pipeline is ready to use!")
        print("Run: python backend/enhanced_pipeline.py --help")
        return 0
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("="*60)
        print("\nPlease review the failures above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
