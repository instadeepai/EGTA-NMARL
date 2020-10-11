regen=0.055
base_args="base_dir=/wd/results/regen_exp/ regen_rate=$regen seed=0 "
tag="/wd/results/regen_exp/tragedy/agent_tags.p"

algs="ia2c,ia2c_fp,ia2c_ind,ma2c_cu,ma2c_ic3,ma2c_dial,ma2c_nc"
defectors="0,1,2,3,4"
runs="20"

python regen_exp.py -m $base_args runs=$runs defectors=$defectors tag_file=$tag algorithm=$algs

