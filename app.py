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
import glob
import shutil
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
try:
    import paramiko
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False
    print("Warning: paramiko not installed. SSH functionality will be disabled.")

app = Flask(__name__)
CORS(app)

# Configuration
RESULTS_DIR = Path("./results")
TEMP_DIR = Path("./temp")
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# GPU Configuration - set the number of available GPUs
NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
#AVAILABLE_GPUS = list(range(NUM_GPUS))
AVAILABLE_GPUS=[2, 0, 3, 1] #did work
#AVAILABLE_GPUS=[3, 0, 2, 1] #did not work
#AVAILABLE_GPUS=[0, 2, 1, 3] #did no work

# Store job results in memory (in production, use Redis or database)
job_results = {}

# Load app configuration
def load_app_data():
    """Load application configuration from app_data.json"""
    try:
        with open("app_data.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: app_data.json not found, using default configuration")
        return {"orchard_config": {"paths": {}, "settings": {}, "boltz_defaults": {}}}

APP_DATA = load_app_data()
ORCHARD_CONFIG = APP_DATA.get("orchard_config", {})
CLUSTER_CONFIG = ORCHARD_CONFIG.get("cluster_config", {})

def generate_job_id() -> str:
    """Generate a short unique job ID."""
    return str(uuid.uuid4())[:8]

class SSHClusterConnection:
    """Manages SSH connections to the cluster."""
    
    def __init__(self):
        self.client = None
        self.sftp = None
        
    def connect(self) -> bool:
        """Establish SSH connection to the cluster."""
        if not SSH_AVAILABLE:
            print("Error: paramiko not available. Install with: pip install paramiko")
            return False
            
        if not CLUSTER_CONFIG.get("use_remote_cluster", False):
            return False
            
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connection parameters
            hostname = CLUSTER_CONFIG.get("ssh_host")
            username = CLUSTER_CONFIG.get("ssh_user")
            key_path = os.path.expanduser(CLUSTER_CONFIG.get("ssh_key_path", "~/.ssh/id_rsa"))
            port = CLUSTER_CONFIG.get("ssh_port", 22)
            timeout = CLUSTER_CONFIG.get("connection_timeout", 30)
            
            if not hostname or not username:
                print("Error: ssh_host and ssh_user must be configured")
                return False
            
            # Try key-based authentication first
            try:
                self.client.connect(
                    hostname=hostname,
                    username=username,
                    key_filename=key_path,
                    port=port,
                    timeout=timeout
                )
                print(f"✅ Connected to {hostname} via SSH key")
            except Exception as key_error:
                print(f"Key auth failed: {key_error}")
                # Fallback to password (will prompt)
                try:
                    import getpass
                    password = getpass.getpass(f"Password for {username}@{hostname}: ")
                    self.client.connect(
                        hostname=hostname,
                        username=username,
                        password=password,
                        port=port,
                        timeout=timeout
                    )
                    print(f"✅ Connected to {hostname} via password")
                except Exception as pass_error:
                    print(f"Password auth failed: {pass_error}")
                    return False
            
            # Set up SFTP
            self.sftp = self.client.open_sftp()
            
            # Ensure remote work directory exists
            remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/tmp/boltz2_api_remote")
            try:
                self.execute_command(f"mkdir -p {remote_work_dir}")
                print(f"✅ Remote work directory ready: {remote_work_dir}")
            except Exception as e:
                print(f"Warning: Could not create remote work directory: {e}")
            
            return True
            
        except Exception as e:
            print(f"SSH connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close SSH connection."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
    
    def execute_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """Execute a command on the remote cluster."""
        if not self.client:
            raise Exception("Not connected to cluster")
        
        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            return exit_code, stdout_text, stderr_text
        except Exception as e:
            raise Exception(f"Command execution failed: {e}")
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to the cluster."""
        if not self.sftp:
            return False
        
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            try:
                self.execute_command(f"mkdir -p {remote_dir}")
            except:
                pass
            
            self.sftp.put(local_path, remote_path)
            return True
        except Exception as e:
            print(f"File upload failed: {e}")
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the cluster."""
        if not self.sftp:
            return False
        
        try:
            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)
            
            self.sftp.get(remote_path, local_path)
            return True
        except Exception as e:
            print(f"File download failed: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the remote cluster."""
        if not self.sftp:
            return False
        
        try:
            self.sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

# Global SSH connection instance
ssh_cluster = SSHClusterConnection()

def get_full_sequence(seq_dict):
    """Get full sequence from sequence dictionary."""
    return ''.join([str(seq_dict[chain_id]) for chain_id in seq_dict.keys()])

def sequence_from_pdb(pdb_path: str):
    """Extract protein sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    ppb = PPBuilder()
    
    sequence_dict = {}
    
    for model in structure:
        for chain in model:
            sequence = ""
            for pp in ppb.build_peptides(chain):
                sequence += pp.get_sequence()
            sequence_dict[chain.id] = sequence
    
    return get_full_sequence(sequence_dict)

def create_orchard_directories(job_id: str) -> Tuple[Path, Path]:
    """Create unique directories for orchard job to avoid conflicts."""
    timestamp = str(int(time.time()))
    pid = str(os.getpid())
    unique_id = f"{timestamp}_{pid}_{job_id}"
    
    base_dir = ORCHARD_CONFIG.get("paths", {}).get("results_base_dir", "/tmp/boltz2_configs")
    
    results_dir = Path(f"{base_dir}/orchard_boltz2_results_{unique_id}")
    temp_dir = Path(f"{base_dir}/orchard_boltz2_temp_{unique_id}")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir, temp_dir

def cleanup_orchard_directories(results_dir: Path, temp_dir: Path):
    """Clean up orchard directories to save disk space."""
    try:
        if results_dir.exists():
            shutil.rmtree(results_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up directories: {results_dir}, {temp_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean directories: {e}")

def append_amides_to_file(amide_smiles: List[str], amide_names: List[str]):
    """Append new amides to the amides file for future reference."""
    amides_file = ORCHARD_CONFIG.get("paths", {}).get("amides_file", "./amides.txt")
    
    # Load existing amides to avoid duplicates
    existing_amides = set()
    if os.path.exists(amides_file):
        try:
            with open(amides_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                        existing_amides.add(line.split(',')[0])
        except Exception as e:
            print(f"Warning: Could not read existing amides file: {e}")
    
    # Append new amides
    try:
        with open(amides_file, 'a') as f:
            for smiles, name in zip(amide_smiles, amide_names):
                if smiles not in existing_amides:
                    f.write(f"{smiles},{name}\n")
                    existing_amides.add(smiles)
                    print(f"📝 Added new amide to database: {name} ({smiles})")
    except Exception as e:
        print(f"Warning: Could not write to amides file: {e}")

def load_pdb_length_data() -> Dict[str, int]:
    """Load PDB length data for filtering."""
    length_file = ORCHARD_CONFIG.get("paths", {}).get("pdb_to_length_file", "./pdb_to_length.json")
    try:
        with open(length_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: PDB length file not found: {length_file}")
        return {}

def filter_pdbs_by_chembl_and_length(target_chembl_ids: List[str]) -> List[str]:
    """Filter PDBs by ChEMBL IDs and protein length."""
    pdb_pattern = ORCHARD_CONFIG.get("paths", {}).get("pdb_files_pattern", "../pdb_files_by_chembl_id/*/*_clean.pdb")
    all_pdbs = glob.glob(pdb_pattern)
    
    # Filter by ChEMBL ID if specified
    if target_chembl_ids:
        target_chembl_set = set(target_chembl_ids)
        filtered_pdbs = []
        for pdb_file in all_pdbs:
            # Extract ChEMBL ID from path: ../pdb_files_by_chembl_id/CHEMBLXXX/...
            chembl_id = pdb_file.split('/')[-2]
            if chembl_id in target_chembl_set:
                filtered_pdbs.append(pdb_file)
        all_pdbs = filtered_pdbs
    
    # Filter by protein length
    pdb_to_length = load_pdb_length_data()
    max_length = ORCHARD_CONFIG.get("settings", {}).get("max_protein_length", 1000)
    
    filtered_pdbs = []
    for pdb_file in all_pdbs:
        if pdb_file in pdb_to_length:
            length = pdb_to_length[pdb_file]
            if length < max_length:
                filtered_pdbs.append(pdb_file)
        else:
            # Include if no length data (conservative approach)
            filtered_pdbs.append(pdb_file)
    
    return filtered_pdbs

def create_slurm_script(job_id: str, num_gpus: int, amides_to_process: List[str], names_to_process: List[str], 
                       target_chembl_ids: List[str], options: Dict[str, Any]) -> Path:
    """Create a SLURM script for the orchard job."""
    
    slurm_config = ORCHARD_CONFIG.get("slurm_config", {})
    
    # Use remote work directory if SSH is enabled
    if CLUSTER_CONFIG.get("use_remote_cluster", False):
        remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/tmp/boltz2_api_remote")
        scripts_dir_path = f"{remote_work_dir}/slurm_scripts"
    else:
        scripts_dir_path = slurm_config.get("scripts_dir", "./slurm_scripts")
        scripts_dir = Path(scripts_dir_path)
        scripts_dir.mkdir(exist_ok=True)
    
    script_path = Path(scripts_dir_path) / f"{job_id}.sh"
    
    # Get conda path - use remote or local
    if CLUSTER_CONFIG.get("use_remote_cluster", False):
        conda_path = CLUSTER_CONFIG.get("remote_conda_path", "~/miniconda3/etc/profile.d/conda.sh")
        conda_env = CLUSTER_CONFIG.get("remote_conda_env", "boltz2")
    else:
        conda_path = slurm_config.get("conda_path", "~/miniconda3/etc/profile.d/conda.sh")
        conda_env = slurm_config.get("conda_env", "boltz2")
    
    # Create the SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --time={slurm_config.get("time_limit", "12:00:00")}
#SBATCH --gpus={num_gpus}
#SBATCH --mem-per-gpu={slurm_config.get("mem_per_gpu", "100G")}
#SBATCH --cpus-per-gpu={slurm_config.get("cpus_per_gpu", 8)}
#SBATCH --partition={slurm_config.get("partition", "preempt")}
#SBATCH --job-name={slurm_config.get("job_name_prefix", "boltz2_api")}_{job_id}
#SBATCH --output={scripts_dir_path}/slurm_{job_id}_%j.out
#SBATCH --error={scripts_dir_path}/slurm_{job_id}_%j.err

echo "Starting SLURM job at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: {job_id}"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Activate conda environment
source {conda_path}
conda activate {conda_env}

# Change to work directory
cd {scripts_dir_path}

# Create job data file
cat > {job_id}_data.json << 'EOF'
{{
    "job_id": "{job_id}",
    "amides_to_process": {json.dumps(amides_to_process)},
    "names_to_process": {json.dumps(names_to_process)},
    "target_chembl_ids": {json.dumps(target_chembl_ids)},
    "options": {json.dumps(options)},
    "num_gpus": {num_gpus}
}}
EOF

# Run the orchard processing script
python slurm_boltz_worker.py {job_id}_data.json

echo "SLURM job finished at $(date)"
"""
    
    # If using remote cluster, create script locally first then upload
    if CLUSTER_CONFIG.get("use_remote_cluster", False):
        # Create local temp script
        local_temp_script = Path(f"./temp_{job_id}.sh")
        with open(local_temp_script, 'w') as f:
            f.write(script_content)
        os.chmod(local_temp_script, 0o755)
        return local_temp_script
    else:
        # Create script locally
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        return script_path

def submit_slurm_job(script_path: Path, job_id: str) -> Optional[str]:
    """Submit a SLURM job and return the job ID."""
    try:
        if CLUSTER_CONFIG.get("use_remote_cluster", False):
            # SSH-based submission
            if not ssh_cluster.client:
                if not ssh_cluster.connect():
                    return None
            
            # Upload the script and worker files
            remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/tmp/boltz2_api_remote")
            remote_script_dir = f"{remote_work_dir}/slurm_scripts"
            remote_script_path = f"{remote_script_dir}/{job_id}.sh"
            remote_worker_path = f"{remote_script_dir}/slurm_boltz_worker.py"
            
            # Ensure remote directory exists
            ssh_cluster.execute_command(f"mkdir -p {remote_script_dir}")
            
            # Upload SLURM script
            if not ssh_cluster.upload_file(str(script_path), remote_script_path):
                print("Failed to upload SLURM script")
                return None
            
            # Upload worker script
            worker_script_path = Path("slurm_boltz_worker.py")
            if worker_script_path.exists():
                if not ssh_cluster.upload_file(str(worker_script_path), remote_worker_path):
                    print("Failed to upload worker script")
                    return None
            else:
                print("Warning: slurm_boltz_worker.py not found locally")
            
            # Make script executable and submit job
            ssh_cluster.execute_command(f"chmod +x {remote_script_path}")
            exit_code, stdout, stderr = ssh_cluster.execute_command(f"cd {remote_script_dir} && sbatch {remote_script_path}")
            
            # Clean up local temp script
            if script_path.name.startswith("temp_"):
                script_path.unlink(missing_ok=True)
            
            if exit_code == 0:
                # Extract job ID from sbatch output
                match = re.search(r'Submitted batch job (\d+)', stdout)
                if match:
                    return match.group(1)
            else:
                print(f"Remote SLURM submission failed: {stderr}")
                return None
                
        else:
            # Local submission
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output (e.g., "Submitted batch job 12345")
                match = re.search(r'Submitted batch job (\d+)', result.stdout)
                if match:
                    return match.group(1)
            else:
                print(f"Local SLURM submission failed: {result.stderr}")
                return None
                
    except Exception as e:
        print(f"Error submitting SLURM job: {e}")
        return None

def get_slurm_job_status(slurm_job_id: str) -> Dict[str, Any]:
    """Get the status of a SLURM job."""
    try:
        if CLUSTER_CONFIG.get("use_remote_cluster", False):
            # SSH-based status check
            if not ssh_cluster.client:
                if not ssh_cluster.connect():
                    return {"status": "CONNECTION_ERROR", "reason": "Could not connect to cluster", "found": False}
            
            exit_code, stdout, stderr = ssh_cluster.execute_command(
                f"squeue -j {slurm_job_id} --format=%T,%R --noheader"
            )
            
            if exit_code == 0 and stdout.strip():
                status_line = stdout.strip()
                parts = status_line.split(',')
                return {
                    "status": parts[0] if parts else "UNKNOWN",
                    "reason": parts[1] if len(parts) > 1 else "",
                    "found": True
                }
            else:
                # Job not found in queue, check if it completed
                return check_completed_slurm_job(slurm_job_id)
        else:
            # Local status check
            result = subprocess.run(
                ["squeue", "-j", slurm_job_id, "--format=%T,%R", "--noheader"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                status_line = result.stdout.strip()
                parts = status_line.split(',')
                return {
                    "status": parts[0] if parts else "UNKNOWN",
                    "reason": parts[1] if len(parts) > 1 else "",
                    "found": True
                }
            else:
                # Job not found in queue, check if it completed
                return check_completed_slurm_job(slurm_job_id)
    except Exception as e:
        return {"status": "ERROR", "reason": str(e), "found": False}

def check_completed_slurm_job(slurm_job_id: str) -> Dict[str, Any]:
    """Check if a SLURM job has completed by looking at sacct."""
    try:
        if CLUSTER_CONFIG.get("use_remote_cluster", False):
            # SSH-based sacct check
            if ssh_cluster.client:
                exit_code, stdout, stderr = ssh_cluster.execute_command(
                    f"sacct -j {slurm_job_id} --format=State --noheader --parsable2"
                )
                
                if exit_code == 0 and stdout.strip():
                    states = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
                    if states:
                        # Get the last state (most recent)
                        final_state = states[-1]
                        return {
                            "status": final_state,
                            "reason": "Job completed",
                            "found": True
                        }
        else:
            # Local sacct check
            result = subprocess.run(
                ["sacct", "-j", slurm_job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                states = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                if states:
                    # Get the last state (most recent)
                    final_state = states[-1]
                    return {
                        "status": final_state,
                        "reason": "Job completed",
                        "found": True
                    }
    except Exception as e:
        pass
    
    return {"status": "NOT_FOUND", "reason": "Job not found in queue or history", "found": False}

def get_slurm_log_files(job_id: str, slurm_job_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find SLURM output and error log files for a job."""
    if CLUSTER_CONFIG.get("use_remote_cluster", False):
        # SSH-based log file access
        remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/tmp/boltz2_api_remote")
        remote_scripts_dir = f"{remote_work_dir}/slurm_scripts"
        
        # Look for log files with pattern: slurm_<job_id>_<slurm_job_id>.out/err
        out_pattern = f"slurm_{job_id}_{slurm_job_id}.out"
        err_pattern = f"slurm_{job_id}_{slurm_job_id}.err"
        
        remote_out_file = f"{remote_scripts_dir}/{out_pattern}"
        remote_err_file = f"{remote_scripts_dir}/{err_pattern}"
        
        # Download log files to local temp directory for processing
        local_logs_dir = Path("./temp_logs")
        local_logs_dir.mkdir(exist_ok=True)
        
        local_out_file = local_logs_dir / out_pattern
        local_err_file = local_logs_dir / err_pattern
        
        out_file = None
        err_file = None
        
        if ssh_cluster.client:
            # Check and download output log
            if ssh_cluster.file_exists(remote_out_file):
                if ssh_cluster.download_file(remote_out_file, str(local_out_file)):
                    out_file = local_out_file
            
            # Check and download error log
            if ssh_cluster.file_exists(remote_err_file):
                if ssh_cluster.download_file(remote_err_file, str(local_err_file)):
                    err_file = local_err_file
        
        return (out_file, err_file)
    else:
        # Local log file access
        scripts_dir = Path(ORCHARD_CONFIG.get("slurm_config", {}).get("scripts_dir", "./slurm_scripts"))
        
        # Look for log files with pattern: slurm_<job_id>_<slurm_job_id>.out/err
        out_pattern = f"slurm_{job_id}_{slurm_job_id}.out"
        err_pattern = f"slurm_{job_id}_{slurm_job_id}.err"
        
        out_file = scripts_dir / out_pattern
        err_file = scripts_dir / err_pattern
        
        return (out_file if out_file.exists() else None, err_file if err_file.exists() else None)

def parse_slurm_logs(out_file: Optional[Path], err_file: Optional[Path], num_lines: int = 20) -> Dict[str, Any]:
    """Parse SLURM log files and extract relevant information."""
    log_info = {
        "has_logs": False,
        "last_lines_out": [],
        "last_lines_err": [],
        "error_detected": False,
        "completion_detected": False,
        "error_messages": [],
        "progress_info": {}
    }
    
    # Parse output log
    if out_file and out_file.exists():
        log_info["has_logs"] = True
        try:
            with open(out_file, 'r') as f:
                lines = f.readlines()
            
            # Get last N lines
            log_info["last_lines_out"] = [line.strip() for line in lines[-num_lines:]]
            
            # Check for completion indicators
            full_text = ''.join(lines).lower()
            if any(phrase in full_text for phrase in [
                "job finished", "completed successfully", "✅", "all batches completed"
            ]):
                log_info["completion_detected"] = True
            
            # Extract progress information
            for line in reversed(lines):
                line_lower = line.lower()
                if "processing pdb" in line_lower:
                    log_info["progress_info"]["current_pdb"] = line.strip()
                    break
                elif "combinations to process" in line_lower:
                    log_info["progress_info"]["total_combinations"] = line.strip()
                elif "starting slurm job" in line_lower or "starting job" in line_lower:
                    log_info["progress_info"]["job_started"] = line.strip()
                    
        except Exception as e:
            log_info["error_messages"].append(f"Could not read output log: {e}")
    
    # Parse error log
    if err_file and err_file.exists():
        log_info["has_logs"] = True
        try:
            with open(err_file, 'r') as f:
                lines = f.readlines()
            
            # Get last N lines
            log_info["last_lines_err"] = [line.strip() for line in lines[-num_lines:]]
            
            # Check for error indicators
            if lines:  # Non-empty error log usually indicates problems
                log_info["error_detected"] = True
                
                # Look for specific error patterns
                full_text = ''.join(lines).lower()
                error_patterns = [
                    "error", "failed", "exception", "traceback", "cuda out of memory",
                    "segmentation fault", "killed", "timeout", "❌"
                ]
                
                for pattern in error_patterns:
                    if pattern in full_text:
                        log_info["error_messages"].append(f"Detected '{pattern}' in error log")
                        break
                        
        except Exception as e:
            log_info["error_messages"].append(f"Could not read error log: {e}")
    
    return log_info

def analyze_job_status(slurm_status: Dict[str, Any], log_info: Dict[str, Any]) -> str:
    """Analyze job status based on SLURM status and log information."""
    slurm_state = slurm_status.get("status", "UNKNOWN")
    
    if slurm_state == "COMPLETED":
        if log_info.get("completion_detected", False):
            return "completed_successfully"
        elif log_info.get("error_detected", False):
            return "completed_with_errors"
        else:
            return "completed_unknown"
    elif slurm_state == "FAILED":
        return "failed"
    elif slurm_state == "CANCELLED":
        return "cancelled"
    elif slurm_state in ["RUNNING", "R"]:
        if log_info.get("error_detected", False):
            return "running_with_errors"
        else:
            return "running"
    elif slurm_state in ["PENDING", "PD"]:
        return "pending"
    elif slurm_state == "NOT_FOUND":
        if log_info.get("completion_detected", False):
            return "completed_successfully"
        else:
            return "unknown"
    else:
        return f"slurm_{slurm_state.lower()}"

def restructure_results_by_amide(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Restructure results to map amide → ChEMBL ID → affinity data."""
    amide_results = {}
    
    for pdb_result in results:
        if "amide_results" not in pdb_result:
            continue
            
        chembl_id = pdb_result.get("chembl_id", "unknown")
        pdb_file = pdb_result.get("pdb_file", "unknown")
        
        for ligand_id, ligand_result in pdb_result["amide_results"].items():
            if "error" in ligand_result:
                continue  # Skip failed predictions
                
            amide_name = ligand_result.get("amide_name", "unknown")
            amide_smiles = ligand_result.get("amide_smiles", "unknown")
            
            # Initialize amide entry if not exists
            if amide_name not in amide_results:
                amide_results[amide_name] = {
                    "amide_smiles": amide_smiles,
                    "chembl_results": {},
                    "total_predictions": 0,
                    "successful_predictions": 0
                }
            
            # Extract affinity data
            affinity_data = ligand_result.get("affinity_data", {})
            confidence_data = ligand_result.get("confidence_data", {})
            
            if affinity_data:
                # Calculate IC50 from affinity prediction
                pred_aff = affinity_data.get("affinity_pred_value")
                ic50_uM = 10**(pred_aff) * 10**3 if pred_aff is not None else None
                
                binding_prob = affinity_data.get("affinity_probability_binary")
                confidence_score = confidence_data.get("confidence_score") if confidence_data else None
                
                # Store structured result
                amide_results[amide_name]["chembl_results"][chembl_id] = {
                    "pdb_file": pdb_file,
                    "affinity_pred_value": pred_aff,
                    "ic50_uM": ic50_uM,
                    "binding_probability": binding_prob,
                    "confidence_score": confidence_score,
                    "duration": ligand_result.get("duration"),
                    "raw_affinity_data": affinity_data,
                    "raw_confidence_data": confidence_data
                }
                
                amide_results[amide_name]["successful_predictions"] += 1
            
            amide_results[amide_name]["total_predictions"] += 1
    
    return amide_results

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

def run_boltz_prediction(config_path: Path, job_id: str, options: Dict[str, Any] = None, device_id: int = 0) -> tuple[bool, str, Dict[str, Any]]:
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
        "--devices", str(device_id),  
        # Use specified device
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
        "--num_workers", "16",
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

def run_single_ligand_prediction(config_file: Path, job_id: str, ligand_id: str, options: Dict[str, Any], device_id: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for a single ligand in parallel processing."""
    
    output_dir = RESULTS_DIR / job_id / ligand_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Running Boltz-2 prediction for ligand {ligand_id} on GPU {device_id}, results will be saved in {output_dir}')
    
    # Set up environment
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64/stubs:{env.get('LD_LIBRARY_PATH', '')}"
    
    # Run boltz command
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", str(device_id),
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
        result['affinity_data'] = None
    
    try:
        structure_files = list(predictions_dir.glob("*.cif")) + list(predictions_dir.glob("*.pdb"))
        structure_file = str(structure_files[0])
        structure_data = open(structure_file, 'r').read()
        result['structure_data'] = structure_data
    except Exception as e:
        print(f'Error parsing structure data: {e}')
        result['structure_data'] = None
    
    try:
        confidence_file = str(list(predictions_dir.glob("confidence*.json"))[0])
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
        result['confidence_data'] = confidence_data
    except Exception as e:
        print(f'Error parsing confidence data: {e}')
        result['confidence_data'] = None
    
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

def run_orchard_ligand_prediction(config_file: Path, job_id: str, ligand_id: str, options: Dict[str, Any], device_id: int, results_dir: Path, temp_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Run Boltz-2 prediction for orchard cluster with custom directories."""
    
    output_dir = results_dir / job_id / ligand_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up environment for orchard cluster
    env = os.environ.copy()
    if ORCHARD_CONFIG.get("settings", {}).get("cuda_visible_devices_env", True):
        env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Get settings from config
    num_workers = ORCHARD_CONFIG.get("settings", {}).get("default_num_workers", 2)
    timeout = ORCHARD_CONFIG.get("settings", {}).get("timeout_per_ligand", 1800)
    
    # Run boltz command with orchard-specific settings
    cmd = [
        "boltz", "predict", str(config_file),
        "--use_msa_server",
        "--out_dir", str(output_dir),
        "--no_kernels",
        "--devices", "1",  # Use 1 device since we set CUDA_VISIBLE_DEVICES
        "--accelerator", "gpu",
        "--output_format", options.get("output_format", "mmcif"),
        "--diffusion_samples", str(options.get("diffusion_samples", 1)),
        "--diffusion_samples_affinity", ORCHARD_CONFIG.get("boltz_defaults", {}).get("diffusion_samples_affinity", "5"),
        "--sampling_steps", str(options.get("sampling_steps", 200)),
        "--recycling_steps", str(options.get("recycling_steps", 3)),
        "--max_parallel_samples", ORCHARD_CONFIG.get("boltz_defaults", {}).get("max_parallel_samples", "1"),
        "--affinity_mw_correction",
        "--write_full_pae",
        "--num_workers", str(num_workers),
    ]
    
    start_time = time.time()
    
    try:
        output = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        if output.returncode == 0:
            result = parse_results(output_dir / f"boltz_results_{job_id}_{ligand_id}", ligand_id, job_id)
            result["duration"] = duration
            return True, ligand_id, result
        else:
            return False, ligand_id, {"error": output.stderr, "duration": duration}
            
    except subprocess.TimeoutExpired:
        return False, ligand_id, {"error": "timeout", "duration": time.time() - start_time}
    except Exception as e:
        return False, ligand_id, {"error": str(e), "duration": time.time() - start_time}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "gpu_config": {
            "num_gpus": NUM_GPUS,
            "available_gpus": AVAILABLE_GPUS
        }
    })

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
    
    # Run prediction on first available GPU
    success, message, results = run_boltz_prediction(config_file, job_id, options, AVAILABLE_GPUS[0])
    
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
    
    # Run predictions in parallel across available GPUs
    start_time = time.time()
    
    # Create GPU assignments for ligands
    gpu_assignments = {}
    for i, ligand_id in enumerate(ligand_ids):
        gpu_id = AVAILABLE_GPUS[i % NUM_GPUS]
        gpu_assignments[ligand_id] = gpu_id
    
    print(f'Processing {len(ligands)} ligands across {NUM_GPUS} GPUs')
    print(f'GPU assignments: {gpu_assignments}')
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            # Submit all jobs with their assigned GPU
            future_to_ligand = {
                executor.submit(run_single_ligand_prediction, config_file, job_id, ligand_id, options, gpu_assignments[ligand_id]): ligand_id
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

@app.route('/predict_orchard', methods=['POST'])
def predict_orchard():
    """Submit orchard cluster predictions with GPU count, specific amides, and ChEMBL ID filtering."""
    slurm_config = ORCHARD_CONFIG.get("slurm_config", {})
    
    if slurm_config.get("use_slurm", False):
        return predict_orchard_slurm()
    else:
        return predict_orchard_direct()

def predict_orchard_slurm():
    """Submit orchard predictions via SLURM job submission."""
    data = request.json
    
    # Validate required parameters
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    num_gpus = data.get("num_gpus")
    if not num_gpus or not isinstance(num_gpus, int) or num_gpus < 1:
        return jsonify({"error": "num_gpus is required and must be a positive integer"}), 400
    
    # Get amide SMILES - can be a single string or list of strings
    amide_smiles = data.get("amide_smiles")
    if not amide_smiles:
        return jsonify({"error": "amide_smiles is required"}), 400
    
    # Get amide names - must match amide_smiles length
    amide_names = data.get("amide_names")
    if not amide_names:
        return jsonify({"error": "amide_names is required"}), 400
    
    # Normalize both to lists
    if isinstance(amide_smiles, str):
        amides_to_process = [amide_smiles]
    elif isinstance(amide_smiles, list):
        amides_to_process = amide_smiles
    else:
        return jsonify({"error": "amide_smiles must be a string or list of strings"}), 400
    
    if isinstance(amide_names, str):
        names_to_process = [amide_names]
    elif isinstance(amide_names, list):
        names_to_process = amide_names
    else:
        return jsonify({"error": "amide_names must be a string or list of strings"}), 400
    
    # Validate that amide_smiles and amide_names have the same length
    if len(amides_to_process) != len(names_to_process):
        return jsonify({"error": f"amide_smiles and amide_names must have the same length. Got {len(amides_to_process)} SMILES and {len(names_to_process)} names"}), 400
    
    # Get ChEMBL IDs - should be a list
    target_chembl_ids = data.get("target_chembl_ids", [])
    if target_chembl_ids and not isinstance(target_chembl_ids, list):
        return jsonify({"error": "target_chembl_ids must be a list"}), 400
    
    # Generate job ID
    job_id = generate_job_id()
    
    try:
        print(f"🚀 Submitting SLURM job {job_id} with {num_gpus} GPUs")
        print(f"   Amides: {len(amides_to_process)}")
        print(f"   Amide names: {names_to_process}")
        print(f"   ChEMBL IDs: {len(target_chembl_ids) if target_chembl_ids else 'All'}")
        
        # Append new amides to the database file
        append_amides_to_file(amides_to_process, names_to_process)
        
        # Load and filter PDBs to validate we have work to do
        filtered_pdbs = filter_pdbs_by_chembl_and_length(target_chembl_ids)
        
        if not filtered_pdbs:
            return jsonify({"error": "No PDBs found matching the criteria"}), 400
        
        print(f"   PDBs to process: {len(filtered_pdbs)}")
        
        # Extract options
        options = {
            "diffusion_samples": data.get("diffusion_samples", ORCHARD_CONFIG.get("boltz_defaults", {}).get("diffusion_samples", 1)),
            "recycling_steps": data.get("recycling_steps", ORCHARD_CONFIG.get("boltz_defaults", {}).get("recycling_steps", 3)),
            "sampling_steps": data.get("sampling_steps", ORCHARD_CONFIG.get("boltz_defaults", {}).get("sampling_steps", 200)),
            "output_format": data.get("output_format", ORCHARD_CONFIG.get("boltz_defaults", {}).get("output_format", "mmcif"))
        }
        
        # Create SLURM script
        script_path = create_slurm_script(job_id, num_gpus, amides_to_process, names_to_process, target_chembl_ids, options)
        
        # Submit SLURM job
        slurm_job_id = submit_slurm_job(script_path, job_id)
        
        if not slurm_job_id:
            return jsonify({
                "job_id": job_id,
                "status": "failed",
                "error": "Failed to submit SLURM job",
                "timestamp": time.time()
            }), 500
        
        # Store job information
        job_info = {
            "job_id": job_id,
            "slurm_job_id": slurm_job_id,
            "status": "submitted",
            "submission_time": time.time(),
            "summary": {
                "num_gpus_requested": num_gpus,
                "amides_to_process": len(amides_to_process),
                "amide_names": names_to_process,
                "pdbs_to_process": len(filtered_pdbs),
                "total_combinations": len(amides_to_process) * len(filtered_pdbs),
                "target_chembl_ids": target_chembl_ids if target_chembl_ids else "all",
                "options": options
            },
            "script_path": str(script_path)
        }
        
        job_results[job_id] = job_info
        
        return jsonify({
            "job_id": job_id,
            "slurm_job_id": slurm_job_id,
            "status": "submitted",
            "message": f"SLURM job {slurm_job_id} submitted successfully",
            "summary": job_info["summary"],
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "timestamp": time.time()
        }), 500

def predict_orchard_direct():
    """Submit orchard cluster predictions with direct execution (original implementation)."""
    data = request.json
    
    # Validate required parameters
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    num_gpus = data.get("num_gpus")
    if not num_gpus or not isinstance(num_gpus, int) or num_gpus < 1:
        return jsonify({"error": "num_gpus is required and must be a positive integer"}), 400
    
    # Get amide SMILES - can be a single string or list of strings
    amide_smiles = data.get("amide_smiles")
    if not amide_smiles:
        return jsonify({"error": "amide_smiles is required"}), 400
    
    # Get amide names - must match amide_smiles length
    amide_names = data.get("amide_names")
    if not amide_names:
        return jsonify({"error": "amide_names is required"}), 400
    
    # Normalize both to lists
    if isinstance(amide_smiles, str):
        amides_to_process = [amide_smiles]
    elif isinstance(amide_smiles, list):
        amides_to_process = amide_smiles
    else:
        return jsonify({"error": "amide_smiles must be a string or list of strings"}), 400
    
    if isinstance(amide_names, str):
        names_to_process = [amide_names]
    elif isinstance(amide_names, list):
        names_to_process = amide_names
    else:
        return jsonify({"error": "amide_names must be a string or list of strings"}), 400
    
    # Validate that amide_smiles and amide_names have the same length
    if len(amides_to_process) != len(names_to_process):
        return jsonify({"error": f"amide_smiles and amide_names must have the same length. Got {len(amides_to_process)} SMILES and {len(names_to_process)} names"}), 400
    
    # Get ChEMBL IDs - should be a list
    target_chembl_ids = data.get("target_chembl_ids", [])
    if target_chembl_ids and not isinstance(target_chembl_ids, list):
        return jsonify({"error": "target_chembl_ids must be a list"}), 400
    
    # Generate job ID and create directories
    job_id = generate_job_id()
    results_dir, temp_dir = create_orchard_directories(job_id)
    
    try:
        print(f"🚀 Starting orchard job {job_id} with {num_gpus} GPUs")
        print(f"   Amides: {len(amides_to_process)}")
        print(f"   Amide names: {names_to_process}")
        print(f"   ChEMBL IDs: {len(target_chembl_ids) if target_chembl_ids else 'All'}")
        
        # Append new amides to the database file
        append_amides_to_file(amides_to_process, names_to_process)
        
        # Load and filter PDBs
        filtered_pdbs = filter_pdbs_by_chembl_and_length(target_chembl_ids)
        
        if not filtered_pdbs:
            return jsonify({"error": "No PDBs found matching the criteria"}), 400
        
        print(f"   PDBs to process: {len(filtered_pdbs)}")
        
        # Extract options
        options = {
            "diffusion_samples": data.get("diffusion_samples", ORCHARD_CONFIG.get("boltz_defaults", {}).get("diffusion_samples", 1)),
            "recycling_steps": data.get("recycling_steps", ORCHARD_CONFIG.get("boltz_defaults", {}).get("recycling_steps", 3)),
            "sampling_steps": data.get("sampling_steps", ORCHARD_CONFIG.get("boltz_defaults", {}).get("sampling_steps", 200)),
            "output_format": data.get("output_format", ORCHARD_CONFIG.get("boltz_defaults", {}).get("output_format", "mmcif"))
        }
        
        # Set up GPU assignments
        available_gpus = list(range(num_gpus))
        
        # Process results
        all_results = []
        total_combinations = len(amides_to_process) * len(filtered_pdbs)
        processed_count = 0
        
        print(f"   Total combinations to process: {total_combinations}")
        
        # Process each PDB
        for pdb_idx, pdb_file in enumerate(filtered_pdbs):
            try:
                print(f"Processing PDB {pdb_idx + 1}/{len(filtered_pdbs)}: {os.path.basename(pdb_file)}")
                
                # Extract protein sequence
                protein_sequence = sequence_from_pdb(pdb_file)
                
                # Create config files for all amides
                config_files = []
                ligand_ids = []
                
                for i, (amide_smiles, amide_name) in enumerate(zip(amides_to_process, names_to_process)):
                    ligand_id = f"L{i+1}"
                    config_file = create_config_file(protein_sequence, amide_smiles, ligand_id, job_id)
                    # Store in temp_dir instead of default TEMP_DIR
                    orchard_config_file = temp_dir / f"{job_id}_{ligand_id}_{pdb_idx}.yaml"
                    shutil.move(str(config_file), str(orchard_config_file))
                    
                    config_files.append(orchard_config_file)
                    ligand_ids.append(ligand_id)
                
                # Run predictions in parallel for this PDB
                gpu_assignments = {ligand_id: available_gpus[i % num_gpus] for i, ligand_id in enumerate(ligand_ids)}
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    future_to_ligand = {
                        executor.submit(
                            run_orchard_ligand_prediction, 
                            config_file, 
                            job_id, 
                            ligand_id, 
                            options, 
                            gpu_assignments[ligand_id],
                            results_dir,
                            temp_dir
                        ): (ligand_id, i) 
                        for i, (config_file, ligand_id) in enumerate(zip(config_files, ligand_ids))
                    }
                    
                    pdb_results = {}
                    for future in concurrent.futures.as_completed(future_to_ligand):
                        ligand_id, amide_idx = future_to_ligand[future]
                        try:
                            success, returned_ligand_id, result_data = future.result()
                            if success:
                                # Add metadata
                                result_data['amide_smiles'] = amides_to_process[amide_idx]
                                result_data['amide_name'] = names_to_process[amide_idx]
                                result_data['pdb_file'] = pdb_file
                                result_data['chembl_id'] = pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                                pdb_results[ligand_id] = result_data
                            else:
                                pdb_results[ligand_id] = {
                                    "error": result_data.get("error", "Unknown error"),
                                    "amide_smiles": amides_to_process[amide_idx],
                                    "amide_name": names_to_process[amide_idx],
                                    "pdb_file": pdb_file,
                                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                                }
                            processed_count += 1
                        except Exception as e:
                            pdb_results[ligand_id] = {
                                "error": str(e),
                                "amide_smiles": amides_to_process[amide_idx],
                                "amide_name": names_to_process[amide_idx],
                                "pdb_file": pdb_file,
                                "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown'
                            }
                            processed_count += 1
                
                # Store results for this PDB
                pdb_result_entry = {
                    "pdb_file": pdb_file,
                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown',
                    "protein_sequence": protein_sequence,
                    "amide_results": pdb_results,
                    "processed_amides": len(amides_to_process)
                }
                all_results.append(pdb_result_entry)
                
                # Clean up config files for this PDB
                for config_file in config_files:
                    config_file.unlink(missing_ok=True)
                
                # Periodic cleanup
                cleanup_frequency = ORCHARD_CONFIG.get("settings", {}).get("cleanup_frequency", 20)
                if (pdb_idx + 1) % cleanup_frequency == 0:
                    print(f"🧹 Performing periodic cleanup after {pdb_idx + 1} PDBs")
                    # Clean results but keep temp for ongoing work
                    if results_dir.exists():
                        for item in results_dir.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item)
                
            except Exception as e:
                print(f"❌ Error processing PDB {pdb_file}: {e}")
                # Add error entry
                error_entry = {
                    "pdb_file": pdb_file,
                    "chembl_id": pdb_file.split('/')[-2] if '/' in pdb_file else 'unknown',
                    "error": str(e),
                    "processed_amides": 0
                }
                all_results.append(error_entry)
                continue
        
        # Compile final results
        end_time = time.time()
        
        final_results = {
            "job_id": job_id,
            "status": "completed",
            "summary": {
                "num_gpus_used": num_gpus,
                "amides_processed": len(amides_to_process),
                "amide_names": names_to_process,
                "pdbs_processed": len(filtered_pdbs),
                "total_combinations": total_combinations,
                "successful_predictions": processed_count,
                "target_chembl_ids": target_chembl_ids if target_chembl_ids else "all",
                "options": options
            },
            "results": all_results,
            "timestamp": time.time()
        }
        
        # Store in job results for retrieval
        job_results[job_id] = final_results
        
        return jsonify(final_results)
        
    except Exception as e:
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "timestamp": time.time()
        }), 500
        
    finally:
        # Clean up directories
        try:
            cleanup_orchard_directories(results_dir, temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up directories: {e}")

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get results for a specific job."""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
    
    job_info = job_results[job_id]
    
    # If this is a SLURM job, update its status
    if "slurm_job_id" in job_info:
        slurm_status = get_slurm_job_status(job_info["slurm_job_id"])
        job_info["slurm_status"] = slurm_status
        
        # Try to load results if job completed
        if slurm_status["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
            results_file = Path(f"./slurm_scripts/{job_id}_results.json")
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        job_results_data = json.load(f)
                    job_info["results"] = job_results_data
                    job_info["status"] = "completed" if slurm_status["status"] == "COMPLETED" else "failed"
                except Exception as e:
                    job_info["error"] = f"Could not load results: {e}"
    
    return jsonify(job_info)

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get enhanced status of a specific job with log analysis."""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
    
    # Get number of log lines to return (default 20)
    num_lines = request.args.get('log_lines', 20, type=int)
    
    job_info = job_results[job_id]
    
    if "slurm_job_id" in job_info:
        slurm_job_id = job_info["slurm_job_id"]
        slurm_status = get_slurm_job_status(slurm_job_id)
        
        # Get and parse SLURM logs
        out_file, err_file = get_slurm_log_files(job_id, slurm_job_id)
        log_info = parse_slurm_logs(out_file, err_file, num_lines)
        
        # Analyze overall job status
        analyzed_status = analyze_job_status(slurm_status, log_info)
        
        # Check for completed results
        structured_results = None
        if analyzed_status in ["completed_successfully", "completed_with_errors", "completed_unknown"]:
            if CLUSTER_CONFIG.get("use_remote_cluster", False):
                # SSH-based results retrieval
                remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/tmp/boltz2_api_remote")
                remote_results_file = f"{remote_work_dir}/slurm_scripts/{job_id}_results.json"
                local_results_file = Path(f"./temp_results/{job_id}_results.json")
                local_results_file.parent.mkdir(exist_ok=True)
                
                if ssh_cluster.client and ssh_cluster.file_exists(remote_results_file):
                    if ssh_cluster.download_file(remote_results_file, str(local_results_file)):
                        try:
                            with open(local_results_file, 'r') as f:
                                raw_results = json.load(f)
                            if "results" in raw_results:
                                structured_results = restructure_results_by_amide(raw_results["results"])
                        except Exception as e:
                            log_info["error_messages"].append(f"Could not parse results: {e}")
            else:
                # Local results retrieval
                results_file = Path(f"./slurm_scripts/{job_id}_results.json")
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            raw_results = json.load(f)
                        if "results" in raw_results:
                            structured_results = restructure_results_by_amide(raw_results["results"])
                    except Exception as e:
                        log_info["error_messages"].append(f"Could not parse results: {e}")
        
        return jsonify({
            "job_id": job_id,
            "slurm_job_id": slurm_job_id,
            "status": analyzed_status,
            "slurm_status": slurm_status["status"],
            "slurm_reason": slurm_status.get("reason", ""),
            "submission_time": job_info.get("submission_time"),
            "summary": job_info.get("summary", {}),
            "logs": {
                "has_logs": log_info["has_logs"],
                "last_output_lines": log_info["last_lines_out"],
                "last_error_lines": log_info["last_lines_err"],
                "error_detected": log_info["error_detected"],
                "completion_detected": log_info["completion_detected"],
                "error_messages": log_info["error_messages"],
                "progress_info": log_info["progress_info"]
            },
            "results": structured_results
        })
    else:
        return jsonify({
            "job_id": job_id,
            "status": job_info.get("status", "unknown"),
            "timestamp": job_info.get("timestamp"),
            "summary": job_info.get("summary", {})
        })

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
    print(f"Starting Boltz-2 API with {NUM_GPUS} GPU(s): {AVAILABLE_GPUS}")
    print(f"Set NUM_GPUS environment variable to configure GPU count (default: 1)")
    app.run(host='0.0.0.0', port=5000, debug=True) 
