#!/bin/bash
set -euo pipefail

# Parse arguments
NO_REPO=false
NO_MOUNT=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-repo)
      NO_REPO=true
      shift
      ;;
    --no-mount)
      NO_MOUNT=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Get the directory where the script is located without changing current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1) Setup linux dependencies
# su -c 'apt-get update && apt-get install -y sudo'
sudo apt-get install -y less nano htop ncdu nvtop lsof rsync btop jq tmux vim direnv psmisc haproxy curl

# Install speedtest
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | sudo bash
sudo apt-get install speedtest

# 2) Setup virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$PATH:~/.local/bin/

# 4) Setup github
email=${1:-"sguo35@gmail.com"}
name=${2:-"sguo35"}
github_url=${3:-""}
git config --global user.email "$email"
git config --global user.name "$name"

# 5) Mount scratch filesystem unless --no-mount is passed
if [ "$NO_MOUNT" = false ]; then
  echo "Setting up /scratch volume group and mounting..."
  sudo apt -y install lvm2
  sudo vgcreate vg0 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1
  sudo lvcreate -n lv_scratch -l 100%FREE vg0
  sudo mkfs.ext4 /dev/mapper/vg0-lv_scratch
  echo '/dev/mapper/vg0-lv_scratch /scratch ext4 defaults 0 0' | sudo tee -a /etc/fstab
  sudo mkdir -p /scratch
  sudo mount -v /scratch
  sudo df -hPT /scratch
  echo "Giving ownership to user ubuntu"
  sudo chown ubuntu:ubuntu /scratch
else
  echo "Skipping scratch volume group setup (--no-mount passed)."
fi


# mount shared NFS
# showmount -e 10.15.14.41 # this is the host ip
# sudo vim /etc/exports
# /scratch 10.15.46.0/24(rw,sync,no_subtree_check,no_root_squash) # this is the client ip
# /scratch 127.0.0.1(rw,sync,no_subtree_check,no_root_squash) # this is self
# sudo exportfs -ra
# sudo mount -t nfs -o vers=3 10.15.14.41:/scratch ~/sky_workdir
# ln -s /scratch ~/sky_workdir # mount locally via symlink

# manually run apt-get install w/sudo on both machines
# sudo chown -R ubuntu:ubuntu ~/sky_workdir
# change python -m uv venv to python3
# bash base_setup.sh

# HF_HOME=/scratch/hf_cache