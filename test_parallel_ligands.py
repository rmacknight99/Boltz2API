#!/usr/bin/env python3
"""
Test script for parallel multi-ligand affinity prediction with Boltz-2.
This approach uses separate YAML files for each ligand and runs them in parallel.
"""

import os
import subprocess
import time
import json
import yaml
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple

def run_single_ligand_prediction(config_file: Path, output_base_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a single ligand."""
    
    ligand_name = config_file.stem
    output_dir = output_base_dir / ligand_name
    
    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run boltz command
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "1",
        "--accelerator", "gpu",
        "--output_format", "mmcif",
        "--diffusion_samples", "1",
        "--diffusion_samples_affinity", "3",
        "--sampling_steps", "200",
        "--recycling_steps", "3",
        "--max_parallel_samples", "1",
        "--affinity_mw_correction",
        "--write_full_pae",
    ]
    
    print(f"🚀 Starting {ligand_name}: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per ligand
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {ligand_name} completed in {duration:.2f} seconds")
            return True, ligand_name, {"duration": duration, "stdout": result.stdout}
        else:
            print(f"❌ {ligand_name} failed after {duration:.2f} seconds")
            print(f"Exit code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, ligand_name, {"duration": duration, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {ligand_name} timed out after 30 minutes")
        return False, ligand_name, {"error": "timeout"}
    except Exception as e:
        print(f"💥 {ligand_name} error: {e}")
        return False, ligand_name, {"error": str(e)}

def run_parallel_ligand_test():
    """Run parallel Boltz-2 affinity predictions for multiple ligands."""
    
    # Configuration
    config_dir = Path("configs/multi_ligand_batch")
    output_base_dir = Path("parallel_ligand_results")
    
    print("=== Boltz-2 Parallel Multi-Ligand Affinity Test ===")
    print(f"Config directory: {config_dir}")
    print(f"Output base directory: {output_base_dir}")
    
    # Find all ligand config files
    config_files = list(config_dir.glob("ligand_*.yaml"))
    
    if not config_files:
        print("❌ No ligand configuration files found!")
        return
    
    print(f"\nFound {len(config_files)} ligand configurations:")
    for config_file in sorted(config_files):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        ligand_id = config['sequences'][1]['ligand']['id']
        ligand_smiles = config['sequences'][1]['ligand']['smiles']
        print(f"  - {config_file.name}: {ligand_id} ({ligand_smiles})")
    
    # Create output directory
    output_base_dir.mkdir(exist_ok=True)
    
    # Run predictions in parallel
    print(f"\n🔄 Running {len(config_files)} predictions in parallel...")
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(config_files))) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_ligand_prediction, config_file, output_base_dir): config_file
            for config_file in config_files
        }
        
        results = []
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            config_file = future_to_config[future]
            try:
                success, ligand_name, info = future.result()
                results.append((success, ligand_name, info))
            except Exception as e:
                print(f"❌ {config_file.name} generated an exception: {e}")
                results.append((False, config_file.stem, {"error": str(e)}))
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Analyze results
    print(f"\n=== Results Summary ===")
    print(f"Total time: {total_duration:.2f} seconds")
    
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n📊 Successful predictions:")
        for success, ligand_name, info in successful:
            print(f"  - {ligand_name}: {info.get('duration', 'N/A'):.2f}s")
    
    if failed:
        print(f"\n💥 Failed predictions:")
        for success, ligand_name, info in failed:
            print(f"  - {ligand_name}: {info.get('error', 'Unknown error')}")
    
    # Analyze output files
    if successful:
        print(f"\n=== Output Analysis ===")
        analyze_parallel_results(output_base_dir, [r[1] for r in successful])
    
    # Performance comparison
    if len(successful) > 1:
        total_sequential_time = sum(info.get('duration', 0) for _, _, info in successful)
        speedup = total_sequential_time / total_duration
        print(f"\n🚀 Performance Insights:")
        print(f"  - Sequential time estimate: {total_sequential_time:.2f}s")
        print(f"  - Parallel time actual: {total_duration:.2f}s")
        print(f"  - Speedup: {speedup:.2f}x")

def analyze_parallel_results(output_base_dir: Path, successful_ligands: List[str]):
    """Analyze the results from parallel predictions."""
    
    print(f"Analyzing results for {len(successful_ligands)} successful predictions...")
    
    all_affinity_data = []
    
    for ligand_name in successful_ligands:
        results_path = output_base_dir / ligand_name / f"boltz_results_{ligand_name}" / "predictions"
        
        if results_path.exists():
            # Find affinity files
            affinity_files = list(results_path.glob("*affinity*.json"))
            structure_files = list(results_path.glob("*.cif"))
            pre_affinity_files = list(results_path.glob("*pre_affinity*.npz"))
            
            print(f"\n{ligand_name}:")
            print(f"  - Affinity files: {len(affinity_files)}")
            print(f"  - Structure files: {len(structure_files)}")
            print(f"  - Pre-affinity files: {len(pre_affinity_files)}")
            
            # Parse affinity data
            for affinity_file in affinity_files:
                try:
                    with open(affinity_file, 'r') as f:
                        data = json.load(f)
                    
                    affinity_data = {
                        'ligand': ligand_name,
                        'affinity_value': data.get('affinity_pred_value', 'N/A'),
                        'binding_probability': data.get('affinity_probability_binary', 'N/A'),
                        'iptm': data.get('iptm', 'N/A'),
                        'ptm': data.get('ptm', 'N/A')
                    }
                    all_affinity_data.append(affinity_data)
                    
                    print(f"    Affinity: {affinity_data['affinity_value']:.3f}" if isinstance(affinity_data['affinity_value'], (int, float)) else f"    Affinity: {affinity_data['affinity_value']}")
                    print(f"    Binding Prob: {affinity_data['binding_probability']:.3f}" if isinstance(affinity_data['binding_probability'], (int, float)) else f"    Binding Prob: {affinity_data['binding_probability']}")
                    
                except Exception as e:
                    print(f"    Error reading {affinity_file.name}: {e}")
    
    # Comparative analysis
    if len(all_affinity_data) > 1:
        print(f"\n🔍 Comparative Analysis:")
        valid_affinities = [d for d in all_affinity_data if isinstance(d['affinity_value'], (int, float))]
        
        if valid_affinities:
            best_affinity = min(valid_affinities, key=lambda x: x['affinity_value'])
            worst_affinity = max(valid_affinities, key=lambda x: x['affinity_value'])
            
            print(f"  - Best affinity: {best_affinity['ligand']} ({best_affinity['affinity_value']:.3f})")
            print(f"  - Worst affinity: {worst_affinity['ligand']} ({worst_affinity['affinity_value']:.3f})")
            
            # Sort by affinity
            sorted_affinities = sorted(valid_affinities, key=lambda x: x['affinity_value'])
            print(f"  - Ranking (best to worst):")
            for i, data in enumerate(sorted_affinities, 1):
                print(f"    {i}. {data['ligand']}: {data['affinity_value']:.3f}")

if __name__ == "__main__":
    run_parallel_ligand_test() 