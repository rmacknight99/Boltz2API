#!/usr/bin/env python3
"""
Setup script for SSH-based cluster connection.
Run this to configure your cluster connection settings.
"""

import json
import os
from pathlib import Path

def setup_cluster_config():
    """Interactive setup for cluster configuration."""
    
    print("🚀 Boltz-2 API SSH Cluster Setup")
    print("=" * 50)
    
    # Load existing config if it exists
    config_file = Path("app_data.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            app_data = json.load(f)
    else:
        app_data = {"orchard_config": {}}
    
    # Ensure structure exists
    if "orchard_config" not in app_data:
        app_data["orchard_config"] = {}
    if "cluster_config" not in app_data["orchard_config"]:
        app_data["orchard_config"]["cluster_config"] = {}
    
    cluster_config = app_data["orchard_config"]["cluster_config"]
    
    # Gather cluster information
    print("\n📡 Cluster Connection Settings:")
    
    ssh_host = input(f"SSH hostname [{cluster_config.get('ssh_host', 'your-cluster.edu')}]: ").strip()
    if ssh_host:
        cluster_config["ssh_host"] = ssh_host
    
    ssh_user = input(f"SSH username [{cluster_config.get('ssh_user', os.getenv('USER', 'your-username'))}]: ").strip()
    if ssh_user:
        cluster_config["ssh_user"] = ssh_user
    
    ssh_key_path = input(f"SSH key path [{cluster_config.get('ssh_key_path', '~/.ssh/id_rsa')}]: ").strip()
    if ssh_key_path:
        cluster_config["ssh_key_path"] = ssh_key_path
    
    ssh_port = input(f"SSH port [{cluster_config.get('ssh_port', 22)}]: ").strip()
    if ssh_port:
        cluster_config["ssh_port"] = int(ssh_port)
    
    print("\n📁 Remote Paths:")
    
    remote_work_dir = input(f"Remote work directory [{cluster_config.get('remote_work_dir', f'/home/{ssh_user or 'your-username'}/boltz2_api_remote')}]: ").strip()
    if remote_work_dir:
        cluster_config["remote_work_dir"] = remote_work_dir
    
    remote_conda_path = input(f"Remote conda path [{cluster_config.get('remote_conda_path', '~/miniconda3/etc/profile.d/conda.sh')}]: ").strip()
    if remote_conda_path:
        cluster_config["remote_conda_path"] = remote_conda_path
    
    remote_conda_env = input(f"Remote conda environment [{cluster_config.get('remote_conda_env', 'boltz2')}]: ").strip()
    if remote_conda_env:
        cluster_config["remote_conda_env"] = remote_conda_env
    
    print("\n⚙️  SLURM Settings:")
    
    if "slurm_config" not in app_data["orchard_config"]:
        app_data["orchard_config"]["slurm_config"] = {}
    
    slurm_config = app_data["orchard_config"]["slurm_config"]
    
    partition = input(f"SLURM partition [{slurm_config.get('partition', 'preempt')}]: ").strip()
    if partition:
        slurm_config["partition"] = partition
    
    time_limit = input(f"Time limit [{slurm_config.get('time_limit', '12:00:00')}]: ").strip()
    if time_limit:
        slurm_config["time_limit"] = time_limit
    
    mem_per_gpu = input(f"Memory per GPU [{slurm_config.get('mem_per_gpu', '100G')}]: ").strip()
    if mem_per_gpu:
        slurm_config["mem_per_gpu"] = mem_per_gpu
    
    cpus_per_gpu = input(f"CPUs per GPU [{slurm_config.get('cpus_per_gpu', 8)}]: ").strip()
    if cpus_per_gpu:
        slurm_config["cpus_per_gpu"] = int(cpus_per_gpu)
    
    # Enable remote cluster
    cluster_config["use_remote_cluster"] = True
    slurm_config["use_slurm"] = True
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(app_data, f, indent=4)
    
    print(f"\n✅ Configuration saved to {config_file}")
    print("\n🔧 Next steps:")
    print("1. Install paramiko: pip install paramiko")
    print("2. Test SSH connection: ssh {ssh_user}@{ssh_host}")
    print("3. Ensure Boltz-2 is installed in the remote conda environment")
    print("4. Start the API: python app.py")
    
    return True

def test_ssh_connection():
    """Test SSH connection to the cluster."""
    print("\n🔍 Testing SSH connection...")
    
    try:
        from app import ssh_cluster
        
        if ssh_cluster.connect():
            print("✅ SSH connection successful!")
            
            # Test basic command
            exit_code, stdout, stderr = ssh_cluster.execute_command("hostname && date")
            if exit_code == 0:
                print(f"✅ Remote hostname: {stdout.strip()}")
            else:
                print(f"⚠️  Command test failed: {stderr}")
            
            ssh_cluster.disconnect()
            return True
        else:
            print("❌ SSH connection failed")
            return False
            
    except ImportError:
        print("❌ paramiko not installed. Run: pip install paramiko")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        setup_cluster_config()
        
        # Ask if user wants to test connection
        test_conn = input("\nTest SSH connection now? [y/N]: ").strip().lower()
        if test_conn == 'y':
            test_ssh_connection()
            
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
    except Exception as e:
        print(f"\nError during setup: {e}")
