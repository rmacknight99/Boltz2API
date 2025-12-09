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
import logging
import warnings
import threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from Bio.PDB import PDBParser, PPBuilder

# Aggressively suppress Paramiko logging and warnings
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=ResourceWarning, module="paramiko")

try:
    import paramiko
    # Disable paramiko's default logging to stderr
    paramiko.util.log_to_file(os.devnull)
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False
    print("Warning: paramiko not installed. SSH functionality will be disabled.")

app = Flask(__name__)
CORS(app)

# SSH Connection Pool
import threading
import queue

class SSHConnectionPool:
    """Thread-safe SSH connection pool to avoid overwhelming IAP tunnel."""
    
    def __init__(self, host_alias, pool_size=5, max_age=300):
        self.host_alias = host_alias
        self.pool_size = pool_size
        self.max_age = max_age  # 5 minutes
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.creation_lock = threading.Lock()  # Serialize connection creation
        self.connection_times = {}  # client -> timestamp
        
    def get_connection(self, timeout=30):
        """Borrow a connection from the pool. Waits if pool is empty."""
        from ssh_config_handler import create_ssh_client_with_config
        
        # First try non-blocking to see if there's a valid cached connection
        try:
            client = self.pool.get_nowait()
            
            # Check if connection is still valid and not too old
            with self.lock:
                created_at = self.connection_times.get(id(client), 0)
                age = time.time() - created_at
                
            if age < self.max_age:
                try:
                    transport = client.get_transport()
                    if transport and transport.is_active():
                        return client
                except:
                    pass
            
            # Connection is stale, close it and create new one
            try:
                client.close()
            except:
                pass
            with self.lock:
                self.connection_times.pop(id(client), None)
                
        except queue.Empty:
            # Pool is empty, check if we're under capacity
            with self.lock:
                current_count = len(self.connection_times)
            
            # If under capacity, create new connection (serialized)
            if current_count < self.pool_size:
                with self.creation_lock:
                    # Double-check count after acquiring lock
                    with self.lock:
                        current_count = len(self.connection_times)
                    if current_count < self.pool_size:
                        client = create_ssh_client_with_config(self.host_alias, banner_timeout=30)
                        with self.lock:
                            self.connection_times[id(client)] = time.time()
                        return client
                    # Someone else created while we waited, try getting from pool
                    try:
                        client = self.pool.get_nowait()
                        return client
                    except queue.Empty:
                        pass
            
            # At capacity - wait for a connection to be returned
            try:
                client = self.pool.get(timeout=timeout)
                # Validate returned connection
                try:
                    transport = client.get_transport()
                    if transport and transport.is_active():
                        return client
                except:
                    pass
                # Connection invalid, close and create new (serialized)
                try:
                    client.close()
                except:
                    pass
                with self.lock:
                    self.connection_times.pop(id(client), None)
            except queue.Empty:
                # Timeout waiting - return error
                raise TimeoutError(f"Timeout waiting for SSH connection after {timeout}s")
        
        # Create new connection (only reached if stale connection was removed) - serialized
        with self.creation_lock:
            client = create_ssh_client_with_config(self.host_alias, banner_timeout=30)
            with self.lock:
                self.connection_times[id(client)] = time.time()
            return client
    
    def return_connection(self, client):
        """Return a connection to the pool."""
        if not client:
            return
            
        try:
            # Check if connection is still valid
            transport = client.get_transport()
            if transport and transport.is_active():
                # Try to return to pool (non-blocking)
                try:
                    self.pool.put_nowait(client)
                    return
                except queue.Full:
                    # Pool is full, close this connection
                    pass
        except:
            pass
        
        # Connection is invalid or pool is full, close it
        try:
            client.close()
        except:
            pass
        with self.lock:
            self.connection_times.pop(id(client), None)
    
    def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                client = self.pool.get_nowait()
                client.close()
            except:
                pass
        with self.lock:
            self.connection_times.clear()

# Global connection pool (initialized after config is loaded)
ssh_connection_pool = None

# Register database query blueprint
try:
    from database_query_endpoints import db_query_bp
    app.register_blueprint(db_query_bp)
except ImportError:
    print("Warning: Could not import database query endpoints")

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
            # Import our SSH config handler
            from ssh_config_handler import create_ssh_client_with_config
            
            # Connection parameters
            hostname = CLUSTER_CONFIG.get("ssh_host")
            username = CLUSTER_CONFIG.get("ssh_user")
            key_path = os.path.expanduser(CLUSTER_CONFIG.get("ssh_key_path", "~/.ssh/id_rsa"))
            port = CLUSTER_CONFIG.get("ssh_port", 22)
            
            if not hostname:
                print("Error: ssh_host must be configured")
                return False
            
            # Try to connect using SSH config (which handles ProxyCommand)
            try:
                self.client = create_ssh_client_with_config(
                    hostname=hostname,
                    username=username,
                    key_filename=key_path,
                    port=port
                )
                print(f"✅ Connected to {hostname} via SSH config")
            except Exception as config_error:
                print(f"SSH config connection failed: {config_error}")
                
                # Fallback to direct connection if SSH config fails
                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Try key-based authentication
                try:
                    self.client.connect(
                        hostname=hostname,
                        username=username,
                        key_filename=key_path,
                        port=port,
                        timeout=30
                    )
                    print(f"✅ Connected to {hostname} via direct connection")
                except paramiko.ssh_exception.SSHException as ssh_error:
                    error_msg = str(ssh_error)
                    # Check if this is a host key verification failure
                    if "Host key for server" in error_msg and "does not match" in error_msg:
                        print(f"⚠️  Host key verification failed, attempting to fix...")
                        from ssh_config_handler import remove_known_host_entry
                        if remove_known_host_entry(hostname):
                            # Retry after removing conflicting host key
                            try:
                                self.client.connect(
                                    hostname=hostname,
                                    username=username,
                                    key_filename=key_path,
                                    port=port,
                                    timeout=30
                                )
                                print(f"✅ Connected to {hostname} after fixing host key")
                            except Exception as retry_error:
                                print(f"Connection failed after host key fix: {retry_error}")
                                return False
                        else:
                            print(f"Direct connection failed: {ssh_error}")
                            return False
                    else:
                        print(f"Direct connection failed: {ssh_error}")
                        return False
                except Exception as direct_error:
                    print(f"Direct connection failed: {direct_error}")
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
        try:
            if self.sftp:
                self.sftp.close()
                self.sftp = None
        except:
            pass
        
        try:
            if self.client:
                # Get the transport to close any proxy processes
                transport = self.client.get_transport()
                if transport:
                    # Close the underlying socket/proxy
                    sock = transport.sock
                    if sock:
                        sock.close()
                    transport.close()
                self.client.close()
                self.client = None
        except:
            pass
    
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
            # If connection is broken, try to reconnect once
            if "SSH session not active" in str(e) or "Broken pipe" in str(e):
                print(f"SSH connection lost, attempting to reconnect...")
                self.disconnect()
                if self.connect():
                    print("✅ Reconnected successfully, retrying command...")
                    stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
                    exit_code = stdout.channel.recv_exit_status()
                    stdout_text = stdout.read().decode('utf-8')
                    stderr_text = stderr.read().decode('utf-8')
                    return exit_code, stdout_text, stderr_text
                else:
                    raise Exception(f"Failed to reconnect: {e}")
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

# Semaphore to limit concurrent SSH connections (prevent overwhelming gcloud IAP tunnels)
# Allow max 5 simultaneous connections
_ssh_semaphore = threading.Semaphore(5)

def execute_remote_command_with_cleanup(command: str, timeout: int = 60) -> Tuple[int, str, str]:
    """
    Execute a command on the remote cluster with automatic connection cleanup.
    This is safer than using the global ssh_cluster for ProxyCommand connections
    to avoid file descriptor leaks.
    Rate-limited to prevent overwhelming the SSH proxy.
    """
    from ssh_config_handler import create_ssh_client_with_config
    
    ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
    if not ssh_host_alias:
        ssh_host_alias = CLUSTER_CONFIG.get("ssh_host")
    
    if not ssh_host_alias:
        raise Exception("SSH host not configured")
    
    # Use semaphore to limit concurrent connections
    with _ssh_semaphore:
        client = None
        try:
            # Create a fresh connection
            client = create_ssh_client_with_config(ssh_host_alias)
            
            # Execute the command
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            return exit_code, stdout_text, stderr_text
            
        except Exception as e:
            raise Exception(f"Command execution failed: {e}")
        finally:
            # Always clean up the connection
            if client:
                try:
                    transport = client.get_transport()
                    if transport:
                        sock = transport.sock
                        if sock:
                            sock.close()
                        transport.close()
                    client.close()
                except:
                    pass

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
            # SSH-based status check with automatic cleanup
            exit_code, stdout, stderr = execute_remote_command_with_cleanup(
                f"squeue -j {slurm_job_id} --format=%T,%R --noheader",
                timeout=30
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
            # SSH-based sacct check with automatic cleanup
            exit_code, stdout, stderr = execute_remote_command_with_cleanup(
                f"sacct -j {slurm_job_id} --format=State --noheader --parsable2",
                timeout=30
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
        # SSH-based log file access with automatic cleanup
        from ssh_config_handler import create_ssh_client_with_config
        
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
        
        # Create temporary SSH connection for file operations
        ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS', CLUSTER_CONFIG.get("ssh_host"))
        if not ssh_host_alias:
            return (None, None)
        
        client = None
        sftp = None
        try:
            client = create_ssh_client_with_config(ssh_host_alias)
            sftp = client.open_sftp()
            
            # Check and download output log
            try:
                sftp.stat(remote_out_file)
                sftp.get(remote_out_file, str(local_out_file))
                out_file = local_out_file
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error downloading output log: {e}")
            
            # Check and download error log
            try:
                sftp.stat(remote_err_file)
                sftp.get(remote_err_file, str(local_err_file))
                err_file = local_err_file
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error downloading error log: {e}")
                
        except Exception as e:
            print(f"SSH connection failed for log files: {e}")
        finally:
            # Clean up connections
            if sftp:
                try:
                    sftp.close()
                except:
                    pass
            if client:
                try:
                    transport = client.get_transport()
                    if transport:
                        sock = transport.sock
                        if sock:
                            sock.close()
                        transport.close()
                    client.close()
                except:
                    pass
        
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

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    """Create SLURM submission scripts for molecule predictions on the orchard cluster.
    
    Optional: Set 'submit_jobs': true to submit the scripts immediately and monitor their progress.
    Accepts both molecule_* and amide_* parameters for backward compatibility.
    Uses protein_ids to specify target proteins (ChEMBL IDs like 'CHEMBL1234' or uploaded PDB IDs).
    """
    from ssh_config_handler import create_ssh_client_with_config
    from slurm_job_monitor import SlurmJobMonitor
    
    data = request.json
    
    # Validate required parameters
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    # Get molecule SMILES - support both old and new parameter names
    molecule_smiles = data.get("molecule_smiles") or data.get("amide_smiles")
    if not molecule_smiles:
        return jsonify({"error": "molecule_smiles (or amide_smiles) is required"}), 400
    
    # Get molecule names - support both old and new parameter names
    molecule_names = data.get("molecule_names") or data.get("amide_names")
    if not molecule_names:
        return jsonify({"error": "molecule_names (or amide_names) is required"}), 400
    
    # Get Protein IDs - should be a list (can be ChEMBL IDs or uploaded PDB IDs)
    protein_ids = data.get("protein_ids") or data.get("chembl_ids", [])  # Support both new and old parameter names    
    
    # Check if we should submit jobs
    submit_jobs = data.get("submit_jobs", False)
    
    # Normalize to lists
    if isinstance(molecule_smiles, str):
        molecules_to_process = [molecule_smiles]
    elif isinstance(molecule_smiles, list):
        molecules_to_process = molecule_smiles
    else:
        return jsonify({"error": "molecule_smiles must be a string or list of strings"}), 400
    
    if isinstance(molecule_names, str):
        names_to_process = [molecule_names]
    elif isinstance(molecule_names, list):
        names_to_process = molecule_names
    else:
        return jsonify({"error": "molecule_names must be a string or list of strings"}), 400
    
    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]
    elif not isinstance(protein_ids, list):
        protein_ids = []
    
    # Validate lengths - only check that molecule SMILES and names match
    if len(molecules_to_process) != len(names_to_process):
        return jsonify({"error": f"molecule_smiles and molecule_names must have the same length. Got {len(molecules_to_process)} SMILES and {len(names_to_process)} names"}), 400
    
    # Protein IDs can be any number - they represent targets to test against, not a 1:1 mapping with molecules
    
    # SLURM configuration
    slurm_config = ORCHARD_CONFIG.get("slurm_config", {})
    partition = slurm_config.get("partition", "preempt")
    time_limit = slurm_config.get("time_limit", "12:00:00")
    mem_per_gpu = slurm_config.get("mem_per_gpu", "100G")
    cpus_per_gpu = slurm_config.get("cpus_per_gpu", 8)
    conda_env = slurm_config.get("conda_env", "boltz2")
    conda_path = slurm_config.get("conda_path", "/project/flame/rmacknig/miniconda3/etc/profile.d/conda.sh")
    
    # Remote directory structure - require environment variables for security
    cluster_config = ORCHARD_CONFIG.get("cluster_config", {})
    remote_work_dir = os.getenv('BOLTZ_2_REMOTE_WORK_DIR')
    if not remote_work_dir:
        # Try to get from config file, but don't provide a default
        remote_work_dir = cluster_config.get("remote_work_dir")
        if not remote_work_dir:
            return jsonify({
                "error": "BOLTZ_2_REMOTE_WORK_DIR environment variable is required",
                "message": "Set BOLTZ_2_REMOTE_WORK_DIR to the remote cluster working directory"
            }), 400
    
    remote_base_dir = os.getenv('BOLTZ_2_REMOTE_BASE_DIR', f"{remote_work_dir}/boltz2_api_remote")
    
    try:
        # Connect to the cluster via SSH
        ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
        if not ssh_host_alias:
            return jsonify({
                "error": "BOLTZ_2_SSH_HOST_ALIAS environment variable is required",
                "message": "Set BOLTZ_2_SSH_HOST_ALIAS to your SSH config host alias"
            }), 400
        
        print(f"🔌 Connecting to cluster via SSH host alias: {ssh_host_alias}...")
        ssh_client = create_ssh_client_with_config(ssh_host_alias)
        sftp = ssh_client.open_sftp()
        
        created_scripts = []
        
        # Generate timestamp once for all files
        timestamp = int(time.time())
        
        # Debug: Print Protein IDs received
        print(f"📋 Protein IDs received: {protein_ids}")
        
        # If Protein IDs provided, write them to a file
        protein_ids_file_path = None
        if protein_ids and len(protein_ids) > 0:
            # Create a unique filename for Protein IDs
            protein_ids_filename = f"protein_ids_custom_{timestamp}.txt"
            protein_ids_dir = f"{remote_work_dir}/protein_id_files"
            protein_ids_file_path = f"{protein_ids_dir}/{protein_ids_filename}"
            
            try:
                # Ensure protein_id_files directory exists
                try:
                    sftp.stat(protein_ids_dir)
                except FileNotFoundError:
                    print(f"Creating protein_id_files directory...")
                    sftp.mkdir(protein_ids_dir)
                
                # Write Protein IDs to file
                with sftp.open(protein_ids_file_path, 'w') as f:
                    for protein_id in protein_ids:
                        f.write(f"{protein_id}\n")
                
                # Verify file was created
                file_info = sftp.stat(protein_ids_file_path)
                print(f"✅ Created Protein IDs file: {protein_ids_filename} (size: {file_info.st_size} bytes)")
                print(f"   Contains {len(protein_ids)} Protein IDs")
                print(f"   Path: {protein_ids_file_path}")
            except Exception as e:
                print(f"❌ Failed to create Protein IDs file: {e}")
                import traceback
                traceback.print_exc()
                protein_ids_file_path = None
        else:
            print(f"📋 No Protein IDs provided, using default file")
        
        # Default to all ChEMBL IDs if none provided or if file creation failed
        if not protein_ids_file_path:
            protein_ids_file_path = f"{remote_work_dir}/protein_id_files/chembl_ids_all.txt"
            print(f"📋 Using default Protein IDs file: {protein_ids_file_path}")
        
        # Create a submission script for each molecule
        for idx, (smiles, name) in enumerate(zip(molecules_to_process, names_to_process)):
            # Create script filename based on molecule name (sanitized)
            safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
            script_name = f"boltz_{safe_name}_{timestamp}_{idx}.sh"
            
            # Create the SLURM script content
            script_content = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --gpus=1
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --partition={partition}
#SBATCH --job-name=boltz2_{safe_name}
#SBATCH --output={remote_base_dir}/logs/{safe_name}_{timestamp}_%j.out
#SBATCH --error={remote_base_dir}/logs/{safe_name}_{timestamp}_%j.err

echo "Starting job at $(date)"
echo "Running on node: $(hostname)"
echo "Molecule: {name}"
echo "SMILES: {smiles}"
echo "Using Protein IDs file: {protein_ids_file_path}"

# Activate conda environment
source {conda_path}
conda activate {conda_env}

# Change to working directory
cd {remote_work_dir}

# Run the prediction
python ./boltz_predict.py 1 \\
    --molecule-smiles '{smiles}' \\
    --molecule-name '{name}' \\
    --protein-file '{protein_ids_file_path}'

echo "Job finished at $(date)"
"""
            
            # Write script to remote directory
            remote_script_path = f"{remote_base_dir}/scripts/{script_name}"
            
            try:
                # Create a temporary file and upload it
                with sftp.open(remote_script_path, 'w') as f:
                    f.write(script_content)
                
                # Make the script executable
                sftp.chmod(remote_script_path, 0o755)
                
                created_scripts.append({
                    "amide_name": name,
                    "amide_smiles": smiles,
                    "protein_id": protein_ids[idx] if idx < len(protein_ids) else None,
                    "script_name": script_name,
                    "script_path": remote_script_path,
                    "output_file_pattern": f"{remote_base_dir}/logs/{safe_name}_{timestamp}_%j.out",
                    "error_file_pattern": f"{remote_base_dir}/logs/{safe_name}_{timestamp}_%j.err",
                    "index": idx + 1
                })
                
                print(f"✅ Created submission script: {script_name}")
                
            except Exception as e:
                print(f"❌ Failed to create script for {name}: {e}")
                created_scripts.append({
                    "amide_name": name,
                    "amide_smiles": smiles,
                    "protein_id": protein_ids[idx] if idx < len(protein_ids) else None,
                    "error": str(e)
                })
        
        # Submit jobs if requested
        submitted_jobs = []
        job_monitor = None
        
        if submit_jobs:
            print("\n🚀 Submitting SLURM jobs...")
            job_monitor = SlurmJobMonitor(ssh_client, remote_base_dir, ssh_host_alias)
            
            for script_info in created_scripts:
                if 'script_path' in script_info:
                    # Prepare job info with actual file paths for monitoring
                    slurm_job_id = job_monitor.submit_job(script_info['script_path'], script_info)
                    
                    if slurm_job_id:
                        # Set actual output/error file paths with job ID
                        script_info['output_file'] = script_info['output_file_pattern'].replace('%j', slurm_job_id)
                        script_info['error_file'] = script_info['error_file_pattern'].replace('%j', slurm_job_id)
                        script_info['slurm_job_id'] = slurm_job_id
                        script_info['submitted'] = True
                        submitted_jobs.append(slurm_job_id)
                    else:
                        script_info['submitted'] = False
                        script_info['submit_error'] = "Failed to submit job"
        
        # Store job monitor in global dict if jobs were submitted
        job_id = generate_job_id()
        if job_monitor and submitted_jobs:
            # Store the monitor for later status checks
            if not hasattr(app, 'job_monitors'):
                app.job_monitors = {}
            app.job_monitors[job_id] = job_monitor
            # Note: The monitor creates fresh SSH connections as needed for status checks
        
        # Generate response
        if submit_jobs:
            response = jsonify({
                "job_id": job_id,
                "status": "jobs_submitted" if submitted_jobs else "submission_failed",
                "message": f"Submitted {len(submitted_jobs)} jobs for monitoring",
                "submitted_job_ids": submitted_jobs,
                "remote_directory": f"{remote_base_dir}/scripts",
                "scripts": created_scripts,
                "timestamp": time.time()
            })
        else:
            response = jsonify({
                "job_id": job_id,
                "status": "scripts_created",
                "message": f"Created {len([s for s in created_scripts if 'script_path' in s])} submission scripts",
                "remote_directory": f"{remote_base_dir}/scripts",
                "scripts": created_scripts,
                "timestamp": time.time()
            })
        
        return response
        
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": f"Failed to create submission scripts: {str(e)}",
            "timestamp": time.time()
        }), 500
    finally:
        # Always clean up SFTP and SSH connections
        # The job monitor creates fresh connections as needed
        if 'sftp' in locals() and sftp:
            try:
                sftp.close()
            except:
                pass
        
        if 'ssh_client' in locals() and ssh_client:
            try:
                transport = ssh_client.get_transport()
                if transport:
                    sock = transport.sock
                    if sock:
                        sock.close()
                    transport.close()
                ssh_client.close()
            except:
                pass


@app.route('/orchard_job_status/<job_id>', methods=['GET'])
def get_orchard_job_status(job_id: str):
    """Get the status of submitted SLURM jobs."""
    if not hasattr(app, 'job_monitors'):
        return jsonify({"error": "No job monitors found"}), 404
    
    monitor = app.job_monitors.get(job_id)
    if not monitor:
        return jsonify({"error": f"Job monitor not found for job_id: {job_id}"}), 404
    
    # Get all job information
    all_jobs = monitor.get_all_jobs()
    
    # Summarize status
    status_summary = {
        'pending': 0,
        'running': 0,
        'completed': 0,
        'failed': 0,
        'other': 0
    }
    
    for slurm_id, job_info in all_jobs.items():
        status = job_info.get('status', 'UNKNOWN')
        if status in ['PENDING', 'PD']:
            status_summary['pending'] += 1
        elif status in ['RUNNING', 'R']:
            status_summary['running'] += 1
        elif status == 'COMPLETED':
            status_summary['completed'] += 1
        elif status in ['FAILED', 'CANCELLED', 'TIMEOUT']:
            status_summary['failed'] += 1
        else:
            status_summary['other'] += 1
    
    return jsonify({
        "job_id": job_id,
        "total_jobs": len(all_jobs),
        "status_summary": status_summary,
        "jobs": all_jobs,
        "timestamp": time.time()
    })


@app.route('/orchard_job_cancel/<job_id>', methods=['POST'])
def cancel_orchard_jobs(job_id: str):
    """Cancel all jobs in a submission."""
    if not hasattr(app, 'job_monitors'):
        return jsonify({"error": "No job monitors found"}), 404
    
    monitor = app.job_monitors.get(job_id)
    if not monitor:
        return jsonify({"error": f"Job monitor not found for job_id: {job_id}"}), 404
    
    # Get specific job IDs to cancel from request, or cancel all
    data = request.json or {}
    slurm_job_ids = data.get('slurm_job_ids', [])
    
    if not slurm_job_ids:
        # Cancel all jobs
        slurm_job_ids = list(monitor.get_all_jobs().keys())
    
    cancelled = []
    failed = []
    
    for slurm_id in slurm_job_ids:
        if monitor.cancel_job(slurm_id):
            cancelled.append(slurm_id)
        else:
            failed.append(slurm_id)
    
    return jsonify({
        "job_id": job_id,
        "cancelled": cancelled,
        "failed_to_cancel": failed,
        "message": f"Cancelled {len(cancelled)} jobs",
        "timestamp": time.time()
    })


@app.route('/available_chembl_ids', methods=['GET'])
def get_available_chembl_ids():
    """Get available ChEMBL IDs from the target info file on the remote cluster."""
    target_info_file = "/project/flame/rmacknig/target_info.json"
    
    ssh_client = None
    try:
        # Connect to cluster to read file
        ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
        if not ssh_host_alias:
            ssh_host_alias = ORCHARD_CONFIG.get("cluster_config", {}).get("ssh_host")
            if not ssh_host_alias:
                return jsonify({"error": "BOLTZ_2_SSH_HOST_ALIAS not configured"}), 400
        
        from ssh_config_handler import create_ssh_client_with_config
        ssh_client = create_ssh_client_with_config(ssh_host_alias)
        
        # Check if file exists
        stdin, stdout, stderr = ssh_client.exec_command(f"test -f {target_info_file} && echo 'exists'")
        if stdout.read().decode().strip() != 'exists':
            return jsonify({
                "error": "Target info file not found on remote cluster",
                "path": target_info_file
            }), 404
        
        # Read the target info from remote
        stdin, stdout, stderr = ssh_client.exec_command(f"cat {target_info_file}")
        target_info_json = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            return jsonify({
                "error": "Error reading target info file",
                "details": error
            }), 500
        
        # Parse JSON
        target_info = json.loads(target_info_json)
        
        # Extract ChEMBL IDs (keys)
        chembl_ids = list(target_info.keys())
        
        # Get optional parameters
        include_details = request.args.get('include_details', 'false').lower() == 'true'
        search_term = request.args.get('search', '')
        limit = request.args.get('limit', type=int)
        
        # Filter by search term if provided
        if search_term:
            chembl_ids = [id for id in chembl_ids if search_term.upper() in id.upper()]
        
        # Sort ChEMBL IDs
        chembl_ids.sort()
        
        # Apply limit if specified
        total_count = len(chembl_ids)
        if limit and limit > 0:
            chembl_ids = chembl_ids[:limit]
        
        # Prepare response
        response_data = {
            "total_count": total_count,
            "returned_count": len(chembl_ids),
            "chembl_ids": chembl_ids
        }
        
        # Include target details if requested
        if include_details:
            details = {}
            for chembl_id in chembl_ids:
                details[chembl_id] = target_info.get(chembl_id, {})
            response_data["target_details"] = details
        
        # Add some statistics
        response_data["statistics"] = {
            "total_targets": len(target_info),
            "search_term": search_term if search_term else None,
            "limited_to": limit if limit else None
        }
        
        return jsonify(response_data)
        
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Failed to parse target info file",
            "message": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Failed to read target info",
            "message": str(e)
        }), 500
    finally:
        # Clean up SSH connection
        if ssh_client:
            try:
                transport = ssh_client.get_transport()
                if transport:
                    sock = transport.sock
                    if sock:
                        sock.close()
                    transport.close()
                ssh_client.close()
            except:
                pass


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
    # Check if it's a regular job
    if job_id in job_results:
        job_info = job_results[job_id]
    # Check if it's an orchard monitored job
    elif hasattr(app, 'job_monitors') and job_id in app.job_monitors:
        # For orchard jobs, return monitoring status instead
        # Results should be retrieved via database query endpoints after completion
        monitor = app.job_monitors[job_id]
        all_jobs = monitor.get_all_jobs()
        
        # Count job statuses
        status_counts = {'completed': 0, 'running': 0, 'pending': 0, 'failed': 0}
        for slurm_id, job_data in all_jobs.items():
            status = job_data.get('status', 'UNKNOWN')
            if status == 'COMPLETED':
                status_counts['completed'] += 1
            elif status in ['RUNNING', 'R']:
                status_counts['running'] += 1
            elif status in ['PENDING', 'PD']:
                status_counts['pending'] += 1
            else:
                status_counts['failed'] += 1
        
        # Determine overall status
        if status_counts['running'] > 0 or status_counts['pending'] > 0:
            overall_status = "processing"
        elif status_counts['completed'] == len(all_jobs):
            overall_status = "completed"
        else:
            overall_status = "partially_completed"
        
        return jsonify({
            "job_id": job_id,
            "type": "orchard_batch",
            "status": overall_status,
            "summary": {
                "total_jobs": len(all_jobs),
                "completed": status_counts['completed'],
                "running": status_counts['running'],
                "pending": status_counts['pending'],
                "failed": status_counts['failed']
            },
            "message": "This is an orchard batch job. Use database query endpoints to retrieve results after completion.",
            "slurm_jobs": all_jobs
        })
    else:
        return jsonify({"error": "Job not found"}), 404
    
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
    # Check if it's a regular job
    if job_id in job_results:
        # Get number of log lines to return (default 20)
        num_lines = request.args.get('log_lines', 20, type=int)
        
        job_info = job_results[job_id]
    # Check if it's an orchard monitored job
    elif hasattr(app, 'job_monitors') and job_id in app.job_monitors:
        monitor = app.job_monitors[job_id]
        all_jobs = monitor.get_all_jobs()
        
        # Create a summary for orchard jobs
        status_summary = {
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'other': 0
        }
        
        for slurm_id, job_data in all_jobs.items():
            status = job_data.get('status', 'UNKNOWN')
            if status in ['PENDING', 'PD']:
                status_summary['pending'] += 1
            elif status in ['RUNNING', 'R']:
                status_summary['running'] += 1
            elif status == 'COMPLETED':
                status_summary['completed'] += 1
            elif status in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                status_summary['failed'] += 1
            else:
                status_summary['other'] += 1
        
        # Return orchard job status in a unified format
        return jsonify({
            "job_id": job_id,
            "status": "orchard_monitoring",
            "type": "orchard_batch",
            "total_jobs": len(all_jobs),
            "status_summary": status_summary,
            "slurm_jobs": all_jobs,
            "timestamp": time.time(),
            "message": "This is an orchard batch job. Use individual SLURM job IDs for detailed logs."
        })
    else:
        return jsonify({"error": "Job not found"}), 404
    
    # Continue with regular job processing for non-orchard jobs...
    
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

@app.route('/deposit_pdb_cluster', methods=['POST'])
def upload_pdb_to_cluster():
    """Upload a PDB file to the cluster in a secure location.
    
    Expected form data:
        - pdb_file: The PDB file to upload
        - pdb_id: A unique identifier for this PDB file
    
    Returns:
        JSON with upload status and remote file path
    """
    try:
        # Check if we have cluster configuration
        if not CLUSTER_CONFIG.get("use_remote_cluster", False):
            return jsonify({
                "error": "Remote cluster not configured",
                "message": "This endpoint requires remote cluster configuration"
            }), 400
        
        # Validate form data
        if 'pdb_file' not in request.files:
            return jsonify({"error": "No PDB file provided"}), 400
        
        if 'pdb_id' not in request.form:
            return jsonify({"error": "No PDB ID provided"}), 400
        
        pdb_file = request.files['pdb_file']
        pdb_id = request.form['pdb_id'].strip()
        
        # Validate file
        if pdb_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate PDB ID (sanitize for security)
        if not pdb_id or not re.match(r'^[a-zA-Z0-9_-]+$', pdb_id):
            return jsonify({
                "error": "Invalid PDB ID",
                "message": "PDB ID must contain only alphanumeric characters, underscores, and hyphens"
            }), 400
        
        # Check file extension
        if not pdb_file.filename.lower().endswith(('.pdb', '.cif')):
            return jsonify({
                "error": "Invalid file type",
                "message": "File must have .pdb or .cif extension"
            }), 400
        
        # Connect to cluster
        if not ssh_cluster.client:
            if not ssh_cluster.connect():
                return jsonify({
                    "error": "Could not connect to cluster",
                    "message": "SSH connection failed"
                }), 500
        
        # Create secure paths - use predictions_base_dir for uploaded PDBs
        predictions_base_dir = ORCHARD_CONFIG.get("paths", {}).get("predictions_base_dir", "/tmp/boltz2_predictions")
        remote_pdb_dir = f"{predictions_base_dir}/uploaded_pdbs"
        
        # Create filename with timestamp for uniqueness
        timestamp = int(time.time())
        original_filename = pdb_file.filename
        file_extension = os.path.splitext(original_filename)[1]
        secure_filename = f"{pdb_id}_{timestamp}{file_extension}"
        remote_file_path = f"{remote_pdb_dir}/{secure_filename}"
        
        # Create temporary local file
        local_temp_dir = Path("./temp_uploads")
        local_temp_dir.mkdir(exist_ok=True)
        local_temp_file = local_temp_dir / secure_filename
        
        try:
            # Save file locally first
            pdb_file.save(str(local_temp_file))
            
            # Verify it's a valid PDB/CIF file by checking first few lines
            with open(local_temp_file, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            # Basic validation for PDB/CIF format
            is_pdb = any(line.startswith(('HEADER', 'ATOM', 'HETATM', 'MODEL')) for line in first_lines)
            is_cif = any(line.startswith(('data_', 'loop_', '_atom_site')) for line in first_lines)
            
            if not (is_pdb or is_cif):
                return jsonify({
                    "error": "Invalid file format",
                    "message": "File does not appear to be a valid PDB or CIF file"
                }), 400
            
            # Ensure remote directory exists
            ssh_cluster.execute_command(f"mkdir -p {remote_pdb_dir}")
            
            # Upload file to cluster
            if not ssh_cluster.upload_file(str(local_temp_file), remote_file_path):
                return jsonify({
                    "error": "Upload failed",
                    "message": "Could not upload file to cluster"
                }), 500
            
            # Verify upload by checking file exists and getting size
            exit_code, stdout, stderr = ssh_cluster.execute_command(f"ls -la {remote_file_path}")
            
            if exit_code != 0:
                return jsonify({
                    "error": "Upload verification failed",
                    "message": "File was not found on cluster after upload"
                }), 500
            
            # Parse file info
            file_info = stdout.strip().split()
            file_size = file_info[4] if len(file_info) > 4 else "unknown"
            
            # Set secure permissions (owner read/write only)
            ssh_cluster.execute_command(f"chmod 600 {remote_file_path}")
            
            return jsonify({
                "status": "success",
                "message": "PDB file uploaded successfully",
                "pdb_id": pdb_id,
                "original_filename": original_filename,
                "remote_path": remote_file_path,
                "secure_filename": secure_filename,
                "file_size": file_size,
                "upload_timestamp": timestamp,
                "file_type": "cif" if is_cif else "pdb"
            })
            
        finally:
            # Clean up local temporary file
            if local_temp_file.exists():
                local_temp_file.unlink()
                
    except Exception as e:
        return jsonify({
            "error": "Upload failed",
            "message": str(e)
        }), 500


@app.route('/cleanup', methods=['POST'])
def cleanup_remote_files():
    """Clean up temporary files from the remote cluster.
    
    Removes files from:
    - chembl_id_files/ (or protein_id_files/)
    - scripts/
    - logs/
    - Any other temporary files specified
    
    Request body (optional):
    {
        "confirm": true,  # Required for safety
        "directories": ["chembl_id_files", "protein_id_files", "scripts", "logs"],  # Optional - defaults to all
        "dry_run": false  # If true, only shows what would be deleted without actually deleting
    }
    """
    try:
        # Check if remote cluster is configured
        if not CLUSTER_CONFIG.get("use_remote_cluster", False):
            return jsonify({
                "error": "Remote cluster not configured",
                "message": "This endpoint requires remote cluster configuration"
            }), 400
        
        data = request.json or {}
        
        # Safety check - require explicit confirmation
        if not data.get("confirm", False):
            return jsonify({
                "error": "Confirmation required",
                "message": "Set 'confirm': true in request body to proceed with cleanup",
                "warning": "This will permanently delete temporary files on the remote cluster"
            }), 400
        
        dry_run = data.get("dry_run", False)
        
        # Get the remote work directory
        remote_work_dir = CLUSTER_CONFIG.get("remote_work_dir", "/home/rmacknig/boltz2_api_remote")
        
        # Default directories to clean
        default_dirs = ["chembl_id_files", "protein_id_files", "scripts", "logs"]
        directories_to_clean = data.get("directories", default_dirs)
        
        # Connect to cluster
        if not ssh_cluster.client:
            if not ssh_cluster.connect():
                return jsonify({
                    "error": "Could not connect to cluster",
                    "message": "SSH connection failed"
                }), 500
        
        cleanup_results = []
        total_files_removed = 0
        total_size_freed = 0
        
        for directory in directories_to_clean:
            dir_path = f"{remote_work_dir}/{directory}"
            
            try:
                # Check if directory exists
                exit_code, stdout, stderr = ssh_cluster.execute_command(f"test -d {dir_path} && echo 'exists'")
                if stdout.strip() != 'exists':
                    cleanup_results.append({
                        "directory": directory,
                        "path": dir_path,
                        "status": "not_found",
                        "message": "Directory does not exist"
                    })
                    continue
                
                # Get file count and size before cleanup
                exit_code, stdout, stderr = ssh_cluster.execute_command(
                    f"find {dir_path} -type f | wc -l"
                )
                file_count = int(stdout.strip()) if exit_code == 0 and stdout.strip() else 0
                
                exit_code, stdout, stderr = ssh_cluster.execute_command(
                    f"du -sb {dir_path} | cut -f1"
                )
                dir_size = int(stdout.strip()) if exit_code == 0 and stdout.strip() else 0
                
                if dry_run:
                    # Dry run - just show what would be deleted
                    exit_code, stdout, stderr = ssh_cluster.execute_command(
                        f"find {dir_path} -type f -name '*' | head -10"
                    )
                    sample_files = stdout.strip().split('\n') if stdout.strip() else []
                    
                    cleanup_results.append({
                        "directory": directory,
                        "path": dir_path,
                        "status": "dry_run",
                        "files_to_delete": file_count,
                        "size_bytes": dir_size,
                        "sample_files": sample_files,
                        "message": f"Would delete {file_count} files ({dir_size} bytes)"
                    })
                else:
                    # Actually clean the directory
                    exit_code, stdout, stderr = ssh_cluster.execute_command(
                        f"find {dir_path} -type f -delete"
                    )
                    
                    if exit_code == 0:
                        cleanup_results.append({
                            "directory": directory,
                            "path": dir_path,
                            "status": "cleaned",
                            "files_removed": file_count,
                            "size_freed_bytes": dir_size,
                            "message": f"Successfully deleted {file_count} files ({dir_size} bytes)"
                        })
                        total_files_removed += file_count
                        total_size_freed += dir_size
                    else:
                        cleanup_results.append({
                            "directory": directory,
                            "path": dir_path,
                            "status": "error",
                            "error": stderr.strip(),
                            "message": f"Failed to clean directory: {stderr.strip()}"
                        })
                        
            except Exception as e:
                cleanup_results.append({
                    "directory": directory,
                    "path": dir_path,
                    "status": "error",
                    "error": str(e),
                    "message": f"Error processing directory: {str(e)}"
                })
        
        # Summary
        successful_cleanups = len([r for r in cleanup_results if r.get("status") == "cleaned"])
        failed_cleanups = len([r for r in cleanup_results if r.get("status") == "error"])
        
        response = {
            "status": "completed" if not dry_run else "dry_run",
            "remote_work_dir": remote_work_dir,
            "summary": {
                "directories_processed": len(directories_to_clean),
                "successful_cleanups": successful_cleanups,
                "failed_cleanups": failed_cleanups,
                "total_files_removed": total_files_removed,
                "total_size_freed_bytes": total_size_freed,
                "total_size_freed_mb": round(total_size_freed / (1024 * 1024), 2)
            },
            "details": cleanup_results,
            "timestamp": time.time()
        }
        
        if dry_run:
            response["message"] = "Dry run completed - no files were actually deleted"
        else:
            response["message"] = f"Cleanup completed: {total_files_removed} files removed, {response['summary']['total_size_freed_mb']} MB freed"
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Cleanup operation failed"
        }), 500


@app.route('/slurm_status/<slurm_job_id>', methods=['GET'])
def get_simple_slurm_status(slurm_job_id):
    """Get the status of a SLURM job directly by running squeue on the cluster.
    
    Args:
        slurm_job_id: The SLURM job ID (e.g., "12345")
    
    Returns:
        JSON with job status, or "finished" if not found in queue
    """
    try:
        if CLUSTER_CONFIG.get("use_remote_cluster", False):
            # SSH-based status check with automatic cleanup
            # Use fresh connection to avoid file descriptor leaks with ProxyCommand
            exit_code, stdout, stderr = execute_remote_command_with_cleanup(
                f"squeue -j {slurm_job_id} --format='%i|%T|%M|%l|%D|%C|%R' --noheader",
                timeout=30
            )
            
            if exit_code == 0 and stdout.strip():
                # Parse the output
                parts = stdout.strip().split('|')
                return jsonify({
                    "slurm_job_id": slurm_job_id,
                    "status": parts[1] if len(parts) > 1 else "UNKNOWN",
                    "elapsed_time": parts[2] if len(parts) > 2 else "",
                    "time_limit": parts[3] if len(parts) > 3 else "",
                    "nodes": parts[4] if len(parts) > 4 else "",
                    "cpus": parts[5] if len(parts) > 5 else "",
                    "reason": parts[6] if len(parts) > 6 else "",
                    "in_queue": True
                })
            else:
                # Job not found in queue, it's finished
                return jsonify({
                    "slurm_job_id": slurm_job_id,
                    "status": "FINISHED",
                    "in_queue": False,
                    "message": "Job not found in queue, likely completed"
                })
        else:
            # Local status check
            result = subprocess.run(
                ["squeue", "-j", slurm_job_id, "--format=%i|%T|%M|%l|%D|%C|%R", "--noheader"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse the output
                parts = result.stdout.strip().split('|')
                return jsonify({
                    "slurm_job_id": slurm_job_id,
                    "status": parts[1] if len(parts) > 1 else "UNKNOWN",
                    "elapsed_time": parts[2] if len(parts) > 2 else "",
                    "time_limit": parts[3] if len(parts) > 3 else "",
                    "nodes": parts[4] if len(parts) > 4 else "",
                    "cpus": parts[5] if len(parts) > 5 else "",
                    "reason": parts[6] if len(parts) > 6 else "",
                    "in_queue": True
                })
            else:
                # Job not found in queue
                return jsonify({
                    "slurm_job_id": slurm_job_id,
                    "status": "FINISHED",
                    "in_queue": False,
                    "message": "Job not found in queue, likely completed"
                })
                
    except Exception as e:
        return jsonify({
            "slurm_job_id": slurm_job_id,
            "status": "ERROR",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize SSH connection pool
    ssh_host_alias = os.getenv('BOLTZ_2_SSH_HOST_ALIAS')
    if ssh_host_alias:
        ssh_connection_pool = SSHConnectionPool(ssh_host_alias, pool_size=5)
        app.config['SSH_POOL'] = ssh_connection_pool
        print(f"✅ Initialized SSH connection pool for '{ssh_host_alias}' with 5 connections")
    
    # Get configuration from environment variables
    host = os.getenv('BOLTZ_2_API_HOST', '0.0.0.0')
    port = int(os.getenv('BOLTZ_2_API_PORT', '5001'))
    debug = os.getenv('BOLTZ_2_API_DEBUG', 'true').lower() == 'true'
    
    print(f"Starting Boltz-2 API with {NUM_GPUS} GPU(s): {AVAILABLE_GPUS}")
    print(f"Set NUM_GPUS environment variable to configure GPU count (default: 1)")
    print(f"API will run on {host}:{port} (debug={debug})")
    print(f"Configure with BOLTZ_2_API_HOST, BOLTZ_2_API_PORT, and BOLTZ_2_API_DEBUG environment variables")
    app.run(host=host, port=port, debug=debug) 
