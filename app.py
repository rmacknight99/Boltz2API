#!/usr/bin/env python3
"""
Flask API for Boltz-2 binding affinity predictions.
"""

import os
import subprocess
import time
import json
import yaml
import uuid
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
RESULTS_DIR = Path("./results")
TEMP_DIR = Path("./temp")
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Store job results in memory (in production, use Redis or database)
job_results = {}

def generate_job_id() -> str:
    """Generate a short unique job ID."""
    return str(uuid.uuid4())[:8]

def create_config_file(protein_sequence: str, ligand_smiles: str, ligand_id: str, job_id: str) -> Path:
    """Create a Boltz-2 configuration file for a single ligand."""
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence
                }
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_smiles
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    }
    
    config_file = TEMP_DIR / f"{job_id}_{ligand_id}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

def run_boltz_prediction(config_path: Path, job_id: str, options: Dict[str, Any] = None) -> tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction."""
    if options is None:
        options = {}
    
    output_dir = RESULTS_DIR / job_id
    output_dir.mkdir(exist_ok=True)
    
    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run boltz command with optimized parameters
    cmd = [
        "boltz", "predict", str(config_path),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        # Use this to avoid CUDA compilation issues
        "--devices", "3",  
        # Use 1 device (can be adjusted based on your GPU setup)
        "--accelerator", "gpu",
        "--output_format", options.get("output_format", "mmcif"), 
        # mmcif is more detailed than pdb
        "--diffusion_samples", str(options.get("diffusion_samples", 1)),
        # Default, but could be increased for better results
        "--diffusion_samples_affinity", "5",
        # Increased for better affinity predictions
        "--sampling_steps", str(options.get("sampling_steps", 200)),
        # Default, good balance of speed/accuracy
        "--recycling_steps", str(options.get("recycling_steps", 3)),
        # Default, good balance
        "--max_parallel_samples", "1",
        # Conservative to avoid GPU memory issues
        "--num_workers", "2",
        # Default, adjust based on CPU cores
        "--affinity_mw_correction",
        # Enable molecular weight correction for affinity
        "--write_full_pae",
        # Include confidence information
    ]
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            return True, "Success", parse_results(output_dir / f"boltz_results_{config_path.stem}")
        else:
            return False, f"Boltz-2 failed: {result.stderr}", {}
            
    except subprocess.TimeoutExpired:
        return False, "Prediction timed out", {}
    except Exception as e:
        return False, f"Error: {str(e)}", {}

def run_single_ligand_prediction(config_file: Path, job_id: str, ligand_id: str, options: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a single ligand in parallel processing."""
    
    output_dir = RESULTS_DIR / job_id / ligand_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Running Boltz-2 prediction for ligand {ligand_id}, results will be saved in {output_dir}')
    
    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run boltz command
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "4",
        "--accelerator", "gpu",
        "--output_format", options.get("output_format", "mmcif"),
        "--diffusion_samples", str(options.get("diffusion_samples", 1)),
        "--diffusion_samples_affinity", "5",
        "--sampling_steps", str(options.get("sampling_steps", 200)),
        "--recycling_steps", str(options.get("recycling_steps", 3)),
        "--max_parallel_samples", "1",
        "--affinity_mw_correction",
        "--write_full_pae",
        "--num_workers", "16",
    ]
    
    start_time = time.time()
    
    # print(f'RUNNING BOLTZ COMMAND:\n{' '.join(cmd)}')
    
    try:
        output = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout per ligand
        )
        
        end_time = time.time()
        duration = end_time - start_time
        if output.returncode == 0:
            result = parse_results(output_dir / f"boltz_results_{job_id}_{ligand_id}", ligand_id, job_id)
            result["duration"] = duration
            return True, ligand_id, result
        else:
            return False, ligand_id, {"error": result.stderr, "duration": duration}
            
    except subprocess.TimeoutExpired:
        return False, ligand_id, {"error": "timeout", "duration": time.time() - start_time}
    except Exception as e:
        return False, ligand_id, {"error": str(e), "duration": time.time() - start_time}

def parse_results(results_dir: Path, ligand_id: str, job_id: str) -> Dict[str, Any]:
    """Parse Boltz-2 results from the output directory."""
    predictions_dir = results_dir / "predictions" / f"{job_id}_{ligand_id}"
    
    result = {}
    
    if not predictions_dir.exists():
        return {"error": "No predictions directory found"}
    
    try:
        affinity_file = str(list(predictions_dir.glob("affinity*.json"))[0])
        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)
        result['affinity_data'] = affinity_data
    except Exception as e:
        print(f'Error parsing affinity data: {e}')
    
    try:
        structure_files = list(predictions_dir.glob("*.cif")) + list(predictions_dir.glob("*.pdb"))
        structure_file = str(structure_files[0])
        structure_data = open(structure_file, 'r').read()
        result['structure_data'] = structure_data
    except Exception as e:
        print(f'Error parsing structure data: {e}')
    
    try:
        confidence_file = str(list(predictions_dir.glob("confidence*.json"))[0])
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
        result['confidence_data'] = confidence_data
    except Exception as e:
        print(f'Error parsing confidence data: {e}')
    
    # try:
    #     pre_affinity_file = str(list(predictions_dir.glob("pre_affinity*.npz"))[0])
    #     pre_affinity_data = np.load(pre_affinity_file)
    #     result['pre_affinity_data'] = pre_affinity_data
    # except Exception as e:
    #     print(f'Error parsing pre-affinity data: {e}')
    
    # try:
    #     pae_file = str(list(predictions_dir.glob("pae*.npz"))[0])
    #     pae_data = np.load(pae_file)
    #     result['pae_data'] = pae_data
    # except Exception as e:
    #     print(f'Error parsing PAE data: {e}')
    
    # try:
    #     pde_file = str(list(predictions_dir.glob("pde*.npz"))[0])
    #     pde_data = np.load(pde_file)
    #     result['pde_data'] = pde_data
    # except Exception as e:
    #     print(f'Error parsing PDE data: {e}')
    
    # try:
    #     plddt_file = str(list(predictions_dir.glob("plddt*.npz"))[0])
    #     plddt_data = np.load(plddt_file)
    #     result['plddt_data'] = plddt_data
    # except Exception as e:
    #     print(f'Error parsing PLDDT data: {e}')
    
    result['data_keys'] = list(result.keys())
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/predict', methods=['POST'])
def predict():
    """Submit a single binding affinity prediction."""
    data = request.json
    
    if not data or not data.get("protein_sequence") or not data.get("ligand_smiles"):
        return jsonify({"error": "protein_sequence and ligand_smiles are required"}), 400
    
    job_id = generate_job_id()
    
    # Create config file
    config_file = create_config_file(
        data["protein_sequence"],
        data["ligand_smiles"],
        "B",
        job_id
    )
    
    # Extract options
    options = {
        "diffusion_samples": data.get("diffusion_samples", 1),
        "recycling_steps": data.get("recycling_steps", 3),
        "sampling_steps": data.get("sampling_steps", 200),
        "output_format": data.get("output_format", "mmcif")
    }
    
    # Run prediction
    success, message, results = run_boltz_prediction(config_file, job_id, options)
    
    # Store results
    job_results[job_id] = {
        "status": "completed" if success else "failed",
        "message": message,
        "results": results,
        "timestamp": time.time(),
        "protein_sequence": data["protein_sequence"],
        "ligand_smiles": data["ligand_smiles"]
    }
    
    # Clean up temp file
    config_file.unlink(missing_ok=True)
    
    if success:
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "results": results
        })
    else:
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "error": message
        }), 500

@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    """Submit parallel binding affinity predictions for multiple ligands."""
    data = request.json
    
    if not data or not data.get("protein_sequence") or not data.get("ligands"):
        return jsonify({"error": "protein_sequence and ligands array are required"}), 400
    
    if not isinstance(data["ligands"], list) or len(data["ligands"]) == 0:
        return jsonify({"error": "ligands must be a non-empty array"}), 400
    
    job_id = generate_job_id()
    protein_sequence = data["protein_sequence"]
    ligands = data["ligands"]
    
    # Extract options
    options = {
        "diffusion_samples": data.get("diffusion_samples", 1),
        "recycling_steps": data.get("recycling_steps", 3),
        "sampling_steps": data.get("sampling_steps", 200),
        "output_format": data.get("output_format", "mmcif")
    }
    
    # Create config files for each ligand
    config_files = []
    ligand_ids = []
    
    for i, ligand in enumerate(ligands):
        if isinstance(ligand, dict):
            ligand_id = ligand.get("id", f"L{i+1}")
            ligand_smiles = ligand.get("smiles")
        else:
            ligand_id = f"L{i+1}"
            ligand_smiles = ligand
        
        if not ligand_smiles:
            return jsonify({"error": f"Invalid ligand format at index {i}"}), 400
        
        config_file = create_config_file(protein_sequence, ligand_smiles, ligand_id, job_id)
        config_files.append(config_file)
        ligand_ids.append(ligand_id)
    
    # Run predictions in parallel
    start_time = time.time()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(config_files))) as executor:
            # Submit all jobs
            future_to_ligand = {
                executor.submit(run_single_ligand_prediction, config_file, job_id, ligand_id, options): ligand_id
                for config_file, ligand_id in zip(config_files, ligand_ids)
            }
            
            results = {}
            successful_ligands = []
            failed_ligands = []
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_ligand):
                ligand_id = future_to_ligand[future]
                try:
                    success, returned_ligand_id, result_data = future.result()
                    if success:
                        results[ligand_id] = result_data
                        successful_ligands.append(ligand_id)
                    else:
                        results[ligand_id] = {"error": result_data.get("error", "Unknown error")}
                        failed_ligands.append(ligand_id)
                except Exception as e:
                    results[ligand_id] = {"error": str(e)}
                    failed_ligands.append(ligand_id)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        
        results['total_duration'] = total_duration
        results['job_id'] = job_id
        results['protein_sequence'] = protein_sequence
        results['ligands'] = ligands
        results['options'] = options
        results['status'] = "completed" if successful_ligands else "failed"
        results['timestamp'] = time.time()
        results['data_keys'] = list(results.keys())
        
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "result": results,
        })
        
    finally:
        # Clean up temp config files
        for config_file in config_files:
            config_file.unlink(missing_ok=True)

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get results for a specific job."""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job_results[job_id])

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs."""
    return jsonify({
        "jobs": [
            {
                "job_id": job_id,
                "status": data["status"],
                "timestamp": data["timestamp"],
                "type": data.get("type", "single_ligand")
            }
            for job_id, data in job_results.items()
        ]
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a result file."""
    # Search for the file in all result directories
    for job_dir in RESULTS_DIR.iterdir():
        if job_dir.is_dir():
            for subdir in job_dir.rglob("*"):
                if subdir.is_dir():
                    file_path = subdir / filename
                    if file_path.exists():
                        return send_file(file_path, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404

@app.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job data."""
    # Remove from memory
    if job_id in job_results:
        del job_results[job_id]
    
    # Remove result files
    job_dir = RESULTS_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    
    return jsonify({"message": f"Job {job_id} cleaned up successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 