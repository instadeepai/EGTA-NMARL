# create a list of jobs using a list of envs and algos
num_gpus=2
num_threads=7
envs=( "tragedy" )
algos=( "ia2c_ind" "ma2c_dial" "ia2c_fp" "ia2c" "ma2c_cu" "ma2c_ic3" "ma2c_nc" )

RESULTS_DIR="optim"
TENSORBOARD_DIR="/wd/${RESULTS_DIR}"

GPUS="--gpus all"
SHM="--shm-size=8gb"
TENSORBOARD_PORT=6006
CONTAINER="baselines-optim"
IMAGE="baselines:optim"
CURRENT_DIR=$( pwd )

PORTS="-p ${TENSORBOARD_PORT}:${TENSORBOARD_PORT}"
BASE_FLAGS="-it --rm --init ${PORTS} -v ${CURRENT_DIR}:/wd -w /wd --name ${CONTAINER}"
RUN_FLAGS="${GPUS} ${SHM} ${BASE_FLAGS}"
DOCKER_RUN="docker run ${RUN_FLAGS} ${IMAGE}"

threads=$( eval echo {0..$(( $num_threads-1 ))} )

# create the root directory to store results
mkdir -p "${RESULTS_DIR}"

is_thread_available () {
    local thread_id=$1
    local thread_busy=$( tmux list-session | grep "optim_${thread_id}" )

    # check if tmux session with this id exists
    if [ -z "${thread_busy}" ]; then
        true
    else
        false
    fi
}

get_available_thread () {
    local thread_id=0

    # loop until an available thread is found
    while true
    do
        # check every thread to see if it is busy
        for thread_id in $threads
        do
            if is_thread_available $thread_id
            then
                echo $thread_id
                return
            fi
        done

        # if all threads are busy, wait 1s and check again
        sleep 1
    done
}

# make a list of commands to run
job_list=$(
    for env in ${envs[@]}
    do
        for algo in ${algos[@]}
        do
            echo "python hyp_optim.py -m env=${env} algorithm=${algo} base_dir=${TENSORBOARD_DIR}"
        done
    done
)

# start the docker container with tensorboard running
tmux new -d -s "tensorboard" "${DOCKER_RUN} tensorboard --logdir=${TENSORBOARD_DIR} --port=${TENSORBOARD_PORT}"

# wait for the docker container to start
container_list=$( docker container list | grep ${CONTAINER} )
while [ -z "${container_list}" ]
do
    sleep 0.1
    container_list=$( docker container list | grep ${CONTAINER} )
done

# give each thread a job as the threads become available
job_count=0

# change IFS to split on new line
SAVE_IFS=$IFS
IFS=$'\n'

for job in ${job_list[@]}
do
    IFS=${SAVE_IFS} # reset IFS
    thread_id=$( get_available_thread )
    gpu_id=$(( $job_count % $num_gpus ))
    (( job_count++ ))

    full_command="docker exec -it -e CUDA_VISIBLE_DEVICES=${gpu_id} ${CONTAINER} ${job}"
    echo "tmux session 'optim_${thread_id}' running '${full_command}'..."

    # ${full_command}
    # exit
    tmux new -d -s "optim_${thread_id}" "${full_command} 2>&1 | tee '${RESULTS_DIR}/optim_${thread_id}.log'"
done

# wait for all threads to complete
busy_threads=$( tmux list-session | grep "optim" )
while ! [ -z "${busy_threads}" ]
do
    sleep 0.1
    busy_threads=$( tmux list-session | grep "optim" )
done

docker stop "${CONTAINER}"
