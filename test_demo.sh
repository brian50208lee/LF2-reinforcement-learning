# model path
model=${1:-LF2_agent/brian/release/1590/lf2_agent}
echo 'load pre-train model path:' $model
# run
bash set_test_demo_env.sh
open F.LF/index.html
python LF2_agent/agent_server.py --load $model