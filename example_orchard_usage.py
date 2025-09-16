#!/usr/bin/env python3
"""
Example usage of the Boltz-2 API predict_orchard endpoint for orchard cluster predictions.
"""

import requests
import json
import time

# API configuration
API_BASE = "http://localhost:5000"

def test_orchard_api():
    """Test the Boltz-2 orchard API with example data."""
    
    # Example request data
    data = {
        "num_gpus": 2,  # Number of GPUs to use
        "amide_smiles": [
            "O=C(NCCCCCCCC)C1=CC=CC=C1",  # N-Octylbenzamide
            "O=C(NC1=CC=CC=C1F)C2=CC=CC=C2"  # N-(2-Fluorophenyl)benzamide
        ],
        "amide_names": [
            "N-Octylbenzamide",
            "N-(2-Fluorophenyl)benzamide"
        ],
        "target_chembl_ids": [
            "CHEMBL1075104",  # Example ChEMBL ID
            "CHEMBL1075105",  # Example ChEMBL ID
            "CHEMBL1075106"   # Example ChEMBL ID
        ],
        # Optional parameters with defaults
        "diffusion_samples": 1,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "output_format": "mmcif"
    }
    
    print("Testing Boltz-2 Orchard API...")
    print(f"Number of GPUs: {data['num_gpus']}")
    print(f"Number of amides: {len(data['amide_smiles'])}")
    print(f"Amide names: {data['amide_names']}")
    print(f"ChEMBL IDs to filter: {data['target_chembl_ids']}")
    print()
    
    # 1. Health check
    print("1. Health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✓ API is healthy")
            health_data = response.json()
            print(f"  GPU config: {health_data.get('gpu_config', {})}")
        else:
            print("✗ API health check failed")
            return
    except Exception as e:
        print(f"✗ Could not connect to API: {e}")
        return
    print()
    
    # 2. Submit orchard prediction
    print("2. Submitting orchard prediction...")
    try:
        response = requests.post(
            f"{API_BASE}/predict_orchard",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=3600  # 1 hour timeout for large jobs
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Orchard prediction completed successfully!")
            print(f"  Job ID: {result.get('job_id')}")
            
            # Print summary
            summary = result.get('summary', {})
            print(f"  Summary:")
            print(f"    - GPUs used: {summary.get('num_gpus_used')}")
            print(f"    - Amides processed: {summary.get('amides_processed')}")
            print(f"    - PDBs processed: {summary.get('pdbs_processed')}")
            print(f"    - Total combinations: {summary.get('total_combinations')}")
            print(f"    - Successful predictions: {summary.get('successful_predictions')}")
            
            # Print some result details
            results = result.get('results', [])
            print(f"  Results: {len(results)} PDB entries")
            
            # Show first few results
            for i, pdb_result in enumerate(results[:3]):
                print(f"    PDB {i+1}: {pdb_result.get('chembl_id')} - {pdb_result.get('processed_amides')} amides")
                if 'amide_results' in pdb_result:
                    for ligand_id, amide_result in pdb_result['amide_results'].items():
                        if 'error' not in amide_result:
                            affinity_data = amide_result.get('affinity_data', {})
                            if affinity_data:
                                binding_prob = affinity_data.get('affinity_probability_binary', 'N/A')
                                print(f"      {ligand_id}: Binding prob = {binding_prob}")
            
            if len(results) > 3:
                print(f"    ... and {len(results) - 3} more PDB results")
            
        else:
            error_data = response.json() if response.content else {}
            print(f"✗ Orchard prediction failed with status {response.status_code}")
            print(f"  Error: {error_data.get('error', 'Unknown error')}")
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out - the job might still be running")
    except Exception as e:
        print(f"✗ Error submitting orchard prediction: {e}")
    
    print()

def test_single_amide_orchard():
    """Test with a single amide and no ChEMBL filtering."""
    
    data = {
        "num_gpus": 1,
        "amide_smiles": "O=C(NCCCCCCCC)C1=CC=CC=C1",  # Single amide as string
        "amide_names": "N-Octylbenzamide",  # Single name as string
        "target_chembl_ids": []  # Empty list means all ChEMBL IDs
    }
    
    print("Testing single amide with all ChEMBL IDs...")
    print(f"Amide: {data['amide_smiles']}")
    print(f"ChEMBL filtering: {'None (all)' if not data['target_chembl_ids'] else data['target_chembl_ids']}")
    
    try:
        response = requests.post(
            f"{API_BASE}/predict_orchard",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=1800  # 30 minutes for smaller job
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Single amide orchard prediction completed!")
            summary = result.get('summary', {})
            print(f"  PDBs processed: {summary.get('pdbs_processed')}")
            print(f"  Successful predictions: {summary.get('successful_predictions')}")
        else:
            error_data = response.json() if response.content else {}
            print(f"✗ Failed: {error_data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()

if __name__ == "__main__":
    print("="*80)
    print("BOLTZ-2 ORCHARD API TEST")
    print("="*80)
    
    # Test full functionality
    test_orchard_api()
    
    print("="*80)
    print("SINGLE AMIDE TEST")
    print("="*80)
    
    # Test simpler case
    test_single_amide_orchard()
    
    print("="*80)
    print("TESTS COMPLETED")
    print("="*80)
