#!/usr/bin/env python3
"""
Example usage of the Boltz-2 API for binding affinity predictions.
"""

import requests
import json
import time

# API configuration
API_BASE = "http://localhost:5000"

def test_api():
    """Test the Boltz-2 API with example data."""
    
    # Example protein sequence and ligand SMILES
    data = {
        "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
        "ligand_smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
    }
    
    print("Testing Boltz-2 API...")
    print(f"Protein sequence length: {len(data['protein_sequence'])}")
    print(f"Ligand SMILES: {data['ligand_smiles']}")
    print()
    
    # 1. Health check
    print("1. Health check...")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        print("✓ API is healthy")
        print(f"  Response: {response.json()}")
    else:
        print("✗ API health check failed")
        return
    print()
    
    # 2. Submit prediction
    print("2. Submitting prediction...")
    response = requests.post(
        f"{API_BASE}/predict",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Prediction submitted successfully!")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")
        
        if "results" in result:
            print("  Results:")
            print(json.dumps(result["results"], indent=2))
            
        return result["job_id"]
    else:
        print("✗ Prediction failed")
        print(f"  Status: {response.status_code}")
        print(f"  Error: {response.json()}")
        return None

def test_api_with_options():
    """Test the API with optional parameters for higher quality results."""
    
    # Example with optional parameters for better quality
    data = {
        "protein_sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
        "ligand_smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "diffusion_samples": 3,    # Higher quality, slower
        "recycling_steps": 5,      # Higher quality, slower
        "sampling_steps": 300,     # Higher quality, slower
        "output_format": "pdb"     # PDB format instead of mmcif
    }
    
    print("Testing Boltz-2 API with high-quality options...")
    print("Note: This will take longer but produce better results")
    print()
    
    response = requests.post(
        f"{API_BASE}/predict",
        headers={"Content-Type": "application/json"},
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ High-quality prediction submitted successfully!")
        print(f"  Job ID: {result['job_id']}")
        return result["job_id"]
    else:
        print("✗ High-quality prediction failed")
        print(f"  Status: {response.status_code}")
        print(f"  Error: {response.json()}")
        return None

def get_job_results(job_id):
    """Get results for a specific job."""
    print(f"3. Getting results for job {job_id}...")
    response = requests.get(f"{API_BASE}/results/{job_id}")
    
    if response.status_code == 200:
        results = response.json()
        print("✓ Results retrieved successfully!")
        print(json.dumps(results, indent=2))
        return results
    else:
        print("✗ Failed to get results")
        print(f"  Status: {response.status_code}")
        print(f"  Error: {response.json()}")
        return None

def list_all_jobs():
    """List all jobs."""
    print("4. Listing all jobs...")
    response = requests.get(f"{API_BASE}/jobs")
    
    if response.status_code == 200:
        jobs = response.json()
        print("✓ Jobs listed successfully!")
        print(json.dumps(jobs, indent=2))
        return jobs
    else:
        print("✗ Failed to list jobs")
        print(f"  Status: {response.status_code}")
        print(f"  Error: {response.json()}")
        return None

def cleanup_job(job_id):
    """Clean up a job."""
    print(f"5. Cleaning up job {job_id}...")
    response = requests.delete(f"{API_BASE}/cleanup/{job_id}")
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Job cleaned up successfully!")
        print(f"  Message: {result['message']}")
    else:
        print("✗ Failed to clean up job")
        print(f"  Status: {response.status_code}")
        print(f"  Error: {response.json()}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Standard test (default parameters)")
    print("2. High-quality test (better results, slower)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\n" + "="*50)
        print("HIGH-QUALITY TEST MODE")
        print("="*50)
        job_id = test_api_with_options()
    else:
        print("\n" + "="*50)
        print("STANDARD TEST MODE")
        print("="*50)
        job_id = test_api()
    
    if job_id:
        print("\n" + "="*50)
        
        # Get results
        results = get_job_results(job_id)
        
        print("\n" + "="*50)
        
        # List all jobs
        list_all_jobs()
        
        print("\n" + "="*50)
        
        # Clean up (optional)
        cleanup_response = input(f"Clean up job {job_id}? (y/n): ")
        if cleanup_response.lower() == 'y':
            cleanup_job(job_id)
    
    print("\nDone!") 