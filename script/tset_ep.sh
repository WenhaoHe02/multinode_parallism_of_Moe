SCRIPT_DIR=$(pwd)/$(dirname "$0")
echo $SCRIPT_DIR

torchrun --standalone --nproc_per_node=8 $SCRIPT_DIR/test_ep.py
