GPUS=--gpus all
SHM=--shm-size=8gb
TENSORBOARD_PORT=6006
PORTS=-p $(TENSORBOARD_PORT):$(TENSORBOARD_PORT)
CONTAINER=baselines-optim

BASE_FLAGS=-it --rm --init $(PORTS) -v $(PWD):/wd -w /wd --name $(CONTAINER)
RUN_FLAGS=$(GPUS) $(SHM) $(BASE_FLAGS)
IMAGE=baselines:optim
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
TENSORBOARD_DIR=/wd/results/regen_exp/tragedy

## default run command
cmd=bash

###
## make run will start bash inside docker
## to run another program you can use:
## >> make run cmd=echo\ "hello"
###
run:
	$(DOCKER_RUN) $(cmd)

play_defect:
	$(DOCKER_RUN) bash play_defectors.sh

tensorboard:
	$(DOCKER_RUN) tensorboard --logdir=$(TENSORBOARD_DIR) --port=$(TENSORBOARD_PORT)

connect_tensorboard:
	docker exec -it $(CONTAINER) tensorboard --bind_all --logdir=$(TENSORBOARD_DIR) --port=$(TENSORBOARD_PORT)

demo:
	$(DOCKER_RUN) python run_module.py --base-dir "$(DEMO_DIR)" evaluate --demo --evaluation-seeds 2500,2501,2502

kill:
	docker kill $(CONTAINER)

build:
	docker build --tag base:latest -f Dockerfile.base .
	docker build --tag $(IMAGE) -f Dockerfile.optim .

tune:
	bash multi_thread_hyp_optim.sh

regen_exp:
	bash regen_exp.sh

process_evaluation_data:
	$(DOCKER_RUN) python -m egta.experiments.process_data --data_type evaluation

process_interaction_data:
	$(DOCKER_RUN) python -m egta.experiments.process_data --data_type interaction

plot_restraint_heat_map:
	$(DOCKER_RUN) python -m egta.experiments.restraint_heat_map_plot

classify_agent_behaviour:
	$(DOCKER_RUN) python -m egta.experiments.classify_agent_behaviour

plot_schelling_diagrams:
	$(DOCKER_RUN) python -m egta.experiments.plot_schelling_diagram
