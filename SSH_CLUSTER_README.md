# SSH Cluster Setup for Boltz-2 API

This guide explains how to set up the Boltz-2 API to work with a remote SLURM cluster via SSH.

## Overview

The SSH-based setup allows you to:
- Run the API on your local machine/server
- Submit jobs to a remote SLURM cluster
- Monitor job progress in real-time
- Retrieve results automatically
- Get structured amide → ChEMBL ID → affinity data

## Prerequisites

### On Your Local Machine:
```bash
pip install paramiko  # For SSH functionality
```

### On the Cluster:
- SSH access with key-based authentication (recommended)
- Boltz-2 installed in a conda environment
- SLURM job scheduler
- Access to the required PDB and data files

## Setup Steps

### 1. Run Interactive Setup
```bash
python setup_ssh_cluster.py
```

This will guide you through configuring:
- SSH connection details (hostname, username, key path)
- Remote work directories
- SLURM parameters (partition, time limits, resources)
- Conda environment settings

### 2. Manual Configuration (Alternative)

Edit `app_data.json`:

```json
{
    "orchard_config": {
        "cluster_config": {
            "use_remote_cluster": true,
            "ssh_host": "your-cluster.edu",
            "ssh_user": "your-username",
            "ssh_key_path": "~/.ssh/id_rsa",
            "ssh_port": 22,
            "remote_work_dir": "/home/your-username/boltz2_api_remote",
            "remote_conda_path": "~/miniconda3/etc/profile.d/conda.sh",
            "remote_conda_env": "boltz2",
            "connection_timeout": 30
        },
        "slurm_config": {
            "use_slurm": true,
            "partition": "preempt",
            "time_limit": "12:00:00",
            "mem_per_gpu": "100G",
            "cpus_per_gpu": 8,
            "job_name_prefix": "boltz2_api"
        }
    }
}
```

### 3. Test SSH Connection
```bash
# Test manually
ssh your-username@your-cluster.edu

# Test via API
python -c "from app import ssh_cluster; ssh_cluster.connect()"
```

## Usage

### Start the API (on your local machine):
```bash
python app.py
```

### Submit a Job:
```python
import requests
import time

# Submit job
response = requests.post("http://localhost:5000/predict_orchard", json={
    "num_gpus": 2,
    "amide_smiles": ["O=C(NCCCCCCCC)C1=CC=CC=C1"],
    "amide_names": ["N-Octylbenzamide"],
    "target_chembl_ids": ["CHEMBL123", "CHEMBL456"]
})

job_id = response.json()["job_id"]
slurm_job_id = response.json()["slurm_job_id"]
print(f"Job submitted: {job_id} (SLURM: {slurm_job_id})")
```

### Monitor Job Progress:
```python
while True:
    status = requests.get(f"http://localhost:5000/job_status/{job_id}")
    data = status.json()
    
    print(f"Status: {data['status']}")
    print(f"SLURM Status: {data['slurm_status']}")
    
    # Show last few log lines
    if data['logs']['last_output_lines']:
        print("Latest output:")
        for line in data['logs']['last_output_lines'][-3:]:
            print(f"  {line}")
    
    # Check if completed
    if data['status'].startswith('completed'):
        print("\n🎉 Job completed!")
        if data['results']:
            print("Results by amide:")
            for amide_name, amide_data in data['results'].items():
                print(f"\n{amide_name}:")
                for chembl_id, result in amide_data['chembl_results'].items():
                    bp = result['binding_probability']
                    ic50 = result['ic50_uM']
                    print(f"  {chembl_id}: BP={bp:.3f}, IC50={ic50:.1f}μM")
        break
    elif data['status'] == 'failed':
        print("❌ Job failed")
        break
        
    time.sleep(30)  # Check every 30 seconds
```

## How It Works

### Job Submission Flow:
1. **API receives request** → validates parameters
2. **Creates SLURM script** → generates job script locally
3. **SSH connection** → connects to cluster
4. **File transfer** → uploads script and worker files
5. **Job submission** → runs `sbatch` remotely
6. **Returns job IDs** → API job ID + SLURM job ID

### Job Monitoring:
1. **Status queries** → SSH to cluster, run `squeue`/`sacct`
2. **Log parsing** → downloads and analyzes SLURM logs
3. **Error detection** → identifies failures, timeouts, CUDA errors
4. **Progress tracking** → shows current PDB being processed

### Results Retrieval:
1. **Completion detection** → monitors for job completion
2. **Results download** → fetches results JSON from cluster
3. **Data restructuring** → converts to amide → ChEMBL → affinity mapping
4. **Clean response** → returns structured data via API

## File Structure

### Local Files:
```
├── app.py                     # Main API server
├── app_data.json             # Configuration
├── slurm_boltz_worker.py     # Worker script (uploaded to cluster)
├── setup_ssh_cluster.py     # Setup script
└── temp_logs/                # Downloaded log files
    └── temp_results/         # Downloaded result files
```

### Remote Files (on cluster):
```
/home/username/boltz2_api_remote/
├── slurm_scripts/
│   ├── abc12345.sh           # SLURM job script
│   ├── abc12345_data.json    # Job parameters
│   ├── slurm_abc12345_*.out  # SLURM output logs
│   ├── slurm_abc12345_*.err  # SLURM error logs
│   ├── abc12345_results.json # Final results
│   └── slurm_boltz_worker.py # Worker script
└── ...
```

## Troubleshooting

### SSH Connection Issues:
```bash
# Test basic SSH
ssh -v your-username@your-cluster.edu

# Check SSH key
ssh-add -l
ssh-add ~/.ssh/id_rsa

# Test from Python
python -c "
from app import ssh_cluster
print('Connecting...')
if ssh_cluster.connect():
    print('✅ Connected')
    exit_code, stdout, stderr = ssh_cluster.execute_command('hostname')
    print(f'Remote hostname: {stdout.strip()}')
else:
    print('❌ Connection failed')
"
```

### SLURM Issues:
```bash
# Check SLURM on cluster
ssh your-cluster.edu
squeue -u $USER
sacct -u $USER --starttime=today
```

### API Issues:
```bash
# Check API logs
python app.py  # Look for SSH connection messages

# Test endpoints
curl http://localhost:5000/health
```

### Common Problems:

1. **"paramiko not installed"**
   ```bash
   pip install paramiko
   ```

2. **"SSH connection failed"**
   - Check hostname/username in config
   - Verify SSH key authentication
   - Test manual SSH connection

3. **"Remote work directory not found"**
   - Directory is created automatically
   - Check SSH user has write permissions

4. **"SLURM submission failed"**
   - Check SLURM partition exists
   - Verify resource limits (GPUs, memory)
   - Check conda environment on cluster

5. **"Job stuck in pending"**
   - Check cluster queue: `squeue -u $USER`
   - May need to wait for resources

## Security Notes

- Use SSH key authentication (more secure than passwords)
- Limit SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Consider SSH config for connection management
- API runs locally - no sensitive data exposed on cluster

## Performance Tips

- Use persistent SSH connections (handled automatically)
- Monitor multiple jobs simultaneously
- Clean up old log/result files periodically
- Use appropriate SLURM resource requests

This setup provides a robust, secure way to leverage cluster resources through a clean API interface!
