RESULTS_DIR="results/regen_exp"
TENSORBOARD_DIR="/wd/${RESULTS_DIR}"

GPUS="--gpus all"
SHM="--shm-size=8gb"
TENSORBOARD_PORT=6006
CONTAINER="baselines-regen-exp"
IMAGE="baselines:optim"
CURRENT_DIR=$( pwd )

PORTS="-p ${TENSORBOARD_PORT}:${TENSORBOARD_PORT}"
BASE_FLAGS="-it --rm --init ${PORTS} -v ${CURRENT_DIR}:/wd -w /wd --name ${CONTAINER}"
RUN_FLAGS="${GPUS} ${SHM} ${BASE_FLAGS}"
DOCKER_RUN="docker run ${RUN_FLAGS} ${IMAGE}"

# create the root directory to store results
mkdir -p "${RESULTS_DIR}"

# specify experiment parameters to sweep over
seeds="0,1,2"
algorithms="ia2c,ia2c_fp,ia2c_ind,ma2c_cu,ma2c_ic3,ma2c_dial,ma2c_nc"
regen_rates="0.03,0.042,0.053,0.065,0.077,0.088,0.1"

# start the docker container with tensorboard running
tmux new -d -s "tensorboard" "${DOCKER_RUN} tensorboard --logdir=${TENSORBOARD_DIR} --port=${TENSORBOARD_PORT}"

# wait for the docker container to start
container_list=$( docker container list | grep ${CONTAINER} )
while [ -z "${container_list}" ]
do
    sleep 0.1
    container_list=$( docker container list | grep ${CONTAINER} )
done
echo "Started docker container and tensorboard..."

# start the regen_rate experiments
docker exec -it ${CONTAINER} python regen_exp.py -m base_dir=${TENSORBOARD_DIR} algorithm=${algorithms} regen_rate=${regen_rates} seed=${seeds} train=true 2>&1 | tee "${RESULTS_DIR}/docker-train.log"
docker exec -it ${CONTAINER} python regen_exp.py -m base_dir=${TENSORBOARD_DIR} algorithm=${algorithms} regen_rate=${regen_rates} seed=${seeds} train=false 2>&1 | tee "${RESULTS_DIR}/docker-eval.log"

echo "Experiment complete. Stopping docker container."

# stop the container after the experiments are done
docker stop "${CONTAINER}"
