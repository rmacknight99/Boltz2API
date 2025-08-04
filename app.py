#!/usr/bin/env python3
"""
Flask API for Boltz-2 binding affinity predictions with optimized multi-GPU support.
"""

import os
import subprocess
import time
import json
import yaml
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from queue import Queue
# import concurrent.futures  # Removed - using sequential processing for GPU safety

app = Flask(__name__)
CORS(app)

# Configuration
RESULTS_DIR = Path("./results")
TEMP_DIR = Path("./temp")
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Store job results in memory (in production, use Redis or database)
job_results = {}

@dataclass
class GPUManager:
    """Manages GPU resources across parallel Boltz processes."""
    
    def __init__(self, total_gpus: int = 4):
        self.total_gpus = total_gpus
        self.gpu_queue = Queue()
        self.gpu_assignments = {}
        self.lock = threading.Lock()
        
        # Initialize available GPUs
        for i in range(total_gpus):
            self.gpu_queue.put(i)
    
    def acquire_gpus(self, num_gpus: int, job_id: str) -> Optional[List[int]]:
        """Acquire GPUs for a job."""
        with self.lock:
            if self.gpu_queue.qsize() < num_gpus:
                return None
            
            gpus = []
            for _ in range(num_gpus):
                gpus.append(self.gpu_queue.get())
            
            self.gpu_assignments[job_id] = gpus
            return gpus
    
    def release_gpus(self, job_id: str):
        """Release GPUs after job completion."""
        with self.lock:
            if job_id in self.gpu_assignments:
                gpus = self.gpu_assignments.pop(job_id)
                for gpu in gpus:
                    self.gpu_queue.put(gpu)

# Initialize GPU manager
gpu_manager = GPUManager(total_gpus=4)

def generate_job_id() -> str:
    """Generate a short unique job ID."""
    return str(uuid.uuid4())[:8]

# Removed create_multi_ligand_config - no longer needed with simplified approach

def create_single_ligand_config(protein_sequence: str, ligand_smiles: str, ligand_id: str, job_id: str) -> Path:
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

def run_boltz_batch_prediction(
    config_path: Path, 
    job_id: str, 
    gpus: List[int], 
    options: Dict[str, Any] = None,
    batch_id: str = "batch"
) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a batch of ligands."""
    if options is None:
        options = {}
    
    output_dir = RESULTS_DIR / job_id / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up environment with simple GPU isolation
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # CRITICAL FIX: Always expect exactly one GPU for this function
    # This avoids all DDP complications in Boltz
    if len(gpus) != 1:
        raise ValueError(f"run_boltz_batch_prediction expects exactly 1 GPU, got {len(gpus)}")
    
    gpu_id = gpus[0]
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🎯 Single GPU mode: GPU {gpu_id} only")
    print(f"🔧 CUDA_VISIBLE_DEVICES: {gpu_id}")
    
    cmd = [
        "boltz", "predict", str(config_path),
        "--use_msa_server",  
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "1",  # Always single device - critical for avoiding DDP
        "--accelerator", "gpu",
        "--output_format", options.get("output_format", "mmcif"),
        "--diffusion_samples", str(options.get("diffusion_samples", 1)),
        "--diffusion_samples_affinity", "5",
        "--sampling_steps", str(options.get("sampling_steps", 200)),
        "--recycling_steps", str(options.get("recycling_steps", 3)),
        "--max_parallel_samples", "1",  # Single sample per GPU
        "--affinity_mw_correction",
        "--write_full_pae",
        "--num_workers", "4",  # Fixed number of workers for single GPU
    ]
    
    start_time = time.time()
    
    try:
        print(f"🚀 Running Boltz prediction on GPU {gpu_id}")
        print(f"📋 Command: {' '.join(cmd)}")
        print(f"🔧 CUDA_VISIBLE_DEVICES: {gpu_id}")
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Debug: Check what actually happened with GPU usage
        print(f"🏁 Boltz process completed. Return code: {result.returncode}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        if result.returncode == 0:
            batch_results = parse_batch_results(output_dir / f"boltz_results_{config_path.stem}", job_id, batch_id)
            batch_results["duration"] = duration
            batch_results["gpus_used"] = gpus
            print(f"✅ Success! Processed batch using GPUs {gpus}")
            return True, "Success", batch_results
        else:
            error_msg = f"Boltz-2 failed: {result.stderr}"
            print(f"❌ Boltz error: {error_msg}")
            return False, error_msg, {"duration": duration, "gpus_used": gpus}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Boltz process timed out after {time.time() - start_time:.2f} seconds")
        return False, "Prediction timed out", {"duration": time.time() - start_time, "gpus_used": gpus}
    except Exception as e:
        print(f"💥 Exception in Boltz process: {str(e)}")
        return False, f"Error: {str(e)}", {"duration": time.time() - start_time, "gpus_used": gpus}

def parse_batch_results(results_dir: Path, job_id: str, batch_id: str) -> Dict[str, Any]:
    """Parse Boltz-2 results from a batch prediction."""
    predictions_dir = results_dir / "predictions"
    
    if not predictions_dir.exists():
        return {"error": "No predictions directory found"}
    
    batch_results = {}
    
    # Find all prediction subdirectories
    for pred_dir in predictions_dir.iterdir():
        if pred_dir.is_dir():
            ligand_id = pred_dir.name.replace(f"{job_id}_", "").replace(f"{batch_id}_", "")
            ligand_results = parse_single_result(pred_dir, ligand_id)
            batch_results[ligand_id] = ligand_results
    
    return batch_results

def parse_single_result(pred_dir: Path, ligand_id: str) -> Dict[str, Any]:
    """Parse results for a single ligand."""
    result = {}
    
    try:
        affinity_files = list(pred_dir.glob("affinity*.json"))
        if affinity_files:
            with open(affinity_files[0], 'r') as f:
                result['affinity_data'] = json.load(f)
    except Exception as e:
        print(f'Error parsing affinity data for {ligand_id}: {e}')
        result['affinity_data'] = None
    
    try:
        structure_files = list(pred_dir.glob("*.cif")) + list(pred_dir.glob("*.pdb"))
        if structure_files:
            with open(structure_files[0], 'r') as f:
                result['structure_data'] = f.read()
    except Exception as e:
        print(f'Error parsing structure data for {ligand_id}: {e}')
        result['structure_data'] = None
    
    try:
        confidence_files = list(pred_dir.glob("confidence*.json"))
        if confidence_files:
            with open(confidence_files[0], 'r') as f:
                result['confidence_data'] = json.load(f)
    except Exception as e:
        print(f'Error parsing confidence data for {ligand_id}: {e}')
        result['confidence_data'] = None
    
    return result

# Removed chunk_ligands - no longer needed with simplified approach

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    available_gpus = gpu_manager.gpu_queue.qsize()
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "available_gpus": available_gpus,
        "total_gpus": gpu_manager.total_gpus
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Submit a single binding affinity prediction."""
    data = request.json
    
    if not data or not data.get("protein_sequence") or not data.get("ligand_smiles"):
        return jsonify({"error": "protein_sequence and ligand_smiles are required"}), 400
    
    job_id = generate_job_id()
    
    # Try to acquire a single GPU
    gpus = gpu_manager.acquire_gpus(1, job_id)
    if gpus is None:
        return jsonify({"error": "No GPUs available, please try again later"}), 503
    
    try:
        # Create config file
        config_file = create_single_ligand_config(
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
        success, message, results = run_boltz_batch_prediction(config_file, job_id, gpus, options, "single")
        
        # Store results
        job_results[job_id] = {
            "status": "completed" if success else "failed",
            "message": message,
            "results": results,
            "timestamp": time.time(),
            "protein_sequence": data["protein_sequence"],
            "ligand_smiles": data["ligand_smiles"],
            "gpus_used": gpus
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
            
    finally:
        # Always release GPUs
        gpu_manager.release_gpus(job_id)

def run_single_ligand_job(protein_sequence: str, ligand: Dict[str, str], options: Dict[str, Any], gpu: int, ligand_job_id: str):
    """Run a single ligand prediction job on one GPU."""
    try:
        # Don't acquire GPU here - it's already managed at chunk level
        # Create config for single ligand
        config_file = create_single_ligand_config(
            protein_sequence, 
            ligand["smiles"], 
            ligand["id"], 
            ligand_job_id
        )
        
        try:
            success, message, results = run_boltz_batch_prediction(
                config_file, ligand_job_id, [gpu], options, "single"
            )
            
            if success:
                # Extract the single ligand result
                if results and ligand["id"] in results:
                    return ligand["id"], results[ligand["id"]]
                else:
                    return ligand["id"], results
            else:
                return ligand["id"], {"error": message}
                
        finally:
            config_file.unlink(missing_ok=True)
            
    except Exception as e:
        return ligand["id"], {"error": str(e)}

@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    """Submit parallel single-ligand predictions - much simpler approach."""
    data = request.json
    
    if not data or not data.get("protein_sequence") or not data.get("ligands"):
        return jsonify({"error": "protein_sequence and ligands array are required"}), 400
    
    if not isinstance(data["ligands"], list) or len(data["ligands"]) == 0:
        return jsonify({"error": "ligands must be a non-empty array"}), 400
    
    job_id = generate_job_id()
    protein_sequence = data["protein_sequence"]
    ligands_input = data["ligands"]
    
    # Normalize ligands to dict format
    ligands = []
    for i, ligand in enumerate(ligands_input):
        if isinstance(ligand, dict):
            ligands.append({
                "id": ligand.get("id", f"L{i+1}"),
                "smiles": ligand.get("smiles")
            })
        else:
            ligands.append({
                "id": f"L{i+1}",
                "smiles": ligand
            })
    
    # Validate ligands
    for i, ligand in enumerate(ligands):
        if not ligand["smiles"]:
            return jsonify({"error": f"Invalid ligand format at index {i}"}), 400
    
    # Extract options
    options = {
        "diffusion_samples": data.get("diffusion_samples", 1),
        "recycling_steps": data.get("recycling_steps", 3),
        "sampling_steps": data.get("sampling_steps", 200),
        "output_format": data.get("output_format", "mmcif")
    }
    
    start_time = time.time()
    
    # SAFE SEQUENTIAL APPROACH: Process one ligand at a time
    # This avoids ALL CUDA driver conflicts while we debug the parallelization
    all_results = {}
    
    print(f"🔄 Processing {len(ligands)} ligands SEQUENTIALLY (safe mode)")
    print(f"📊 This will be slower but won't crash your GPUs")
    
    # Process each ligand individually
    for i, ligand in enumerate(ligands):
        print(f"🎯 Processing ligand {i+1}/{len(ligands)}: {ligand['id']}")
        
        # Acquire a single GPU for this ligand
        ligand_job_id = f"{job_id}_ligand_{i}"
        gpus = gpu_manager.acquire_gpus(1, ligand_job_id)
        
        if gpus is None:
            print(f"⚠️  No GPUs available for ligand {ligand['id']}")
            all_results[ligand["id"]] = {"error": "No GPUs available"}
            continue
        
        try:
            gpu_id = gpus[0]
            print(f"✅ Using GPU {gpu_id} for ligand {ligand['id']}")
            
            # Run the ligand prediction directly (no threading)
            ligand_id, result = run_single_ligand_job(
                protein_sequence, ligand, options, gpu_id, ligand_job_id
            )
            all_results[ligand_id] = result
            print(f"✅ Completed ligand {ligand_id}")
            
        except Exception as e:
            print(f"❌ Error processing ligand {ligand['id']}: {e}")
            all_results[ligand["id"]] = {"error": f"Processing failed: {str(e)}"}
            
        finally:
            # Always release GPU immediately after each ligand
            gpu_manager.release_gpus(ligand_job_id)
            print(f"🔄 Released GPU {gpu_id}")
        
        # Small delay between ligands for clean GPU transitions
        if i < len(ligands) - 1:
            print("⏱️  Brief pause before next ligand...")
            time.sleep(1)
    
    end_time = time.time()
    
    final_results = {
        "job_id": job_id,
        "status": "completed",
        "total_duration": end_time - start_time,
        "protein_sequence": protein_sequence,
        "ligands": ligands_input,
        "options": options,
        "results": all_results,
        "strategy": "sequential_safe_mode",
        "max_parallel": 1,
        "num_ligands": len(ligands),
        "devices_per_process": 1
    }
    
    job_results[job_id] = final_results
    
    return jsonify(final_results)

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
                "strategy": data.get("strategy", "single_ligand"),
                "gpus_used": data.get("gpus_used", [])
            }
            for job_id, data in job_results.items()
        ]
    })

@app.route('/gpu_status', methods=['GET'])
def gpu_status():
    """Get current GPU usage status."""
    return jsonify({
        "total_gpus": gpu_manager.total_gpus,
        "available_gpus": gpu_manager.gpu_queue.qsize(),
        "active_assignments": list(gpu_manager.gpu_assignments.keys()),
        "gpu_assignments": gpu_manager.gpu_assignments
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
    # Release any held GPUs
    gpu_manager.release_gpus(job_id)
    
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
    app.run(host='0.0.0.0', port=5000, debug=False) 