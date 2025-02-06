cleanup() {
    echo "Terminating tendermint..."
    if kill -0 "$tm_subprocess_pid" 2>/dev/null; then
        kill "$tm_subprocess_pid"
        wait "$tm_subprocess_pid" 2>/dev/null
    fi
    echo "Tendermint terminated"
}

# Load env vars
source .env
repo_path=$PWD

# Link cleanup to the exit signal
trap cleanup EXIT

# Remove previous agent if exists
if test -d learning_agent; then
  echo "Removing previous agent build"
  rm -r learning_agent
fi

# Remove empty directories to avoid wrong hashes
find . -empty -type d -delete

# Ensure that third party packages are correctly synced
make clean
echo "Fetching autonomy version..."
AUTONOMY_VERSION=v$(autonomy --version | awk '{print $3}')
echo "Autonomy version: $AUTONOMY_VERSION"

echo "Fetching AEA version..."
AEA_VERSION=v$(aea --version | awk '{print $3}')
echo "AEA version: $AEA_VERSION"
autonomy packages sync --source valory-xyz/open-aea:$AEA_VERSION --source valory-xyz/open-autonomy:$AUTONOMY_VERSION --update-packages

# Ensure hashes are updated
autonomy packages lock

# Fetch the agent
autonomy fetch --local --agent valory/learning_agent

# Activate Poetry environment
source $(poetry env info --path)/bin/activate

# Replace params with env vars
source .env
python scripts/aea-config-replace.py

# Copy and add the keys and issue certificates
cd learning_agent
cp $PWD/../ethereum_private_key.txt .
autonomy add-key ethereum ethereum_private_key.txt
autonomy issue-certificates

# Run tendermint
rm -r ~/.tendermint
tendermint init > /dev/null 2>&1
echo "Starting Tendermint..."
tendermint node --proxy_app=tcp://127.0.0.1:26658 --rpc.laddr=tcp://127.0.0.1:26657 --p2p.laddr=tcp://0.0.0.0:26656 --p2p.seeds= --consensus.create_empty_blocks=true > /dev/null 2>&1 &
tm_subprocess_pid=$!

# Run the agent
aea -s run