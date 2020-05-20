GPU = "0,1"  # split by ,
C_NAME = "speech2text_cont"
RAM_LIMIT = "32g"
I_NAME = "speech2text_img"
HOST_SHARED_DIR = "/home/rwerner/SpeechProcessing/shared"
HOST_SHARED_DIR_2 = "/data/rwerner"

BLD_CTX = ".."

build-nocache:
	cp ./Dockerfile ../Dockerfile
	docker build --no-cache -t "$(I_NAME)" -f ../Dockerfile "${BLD_CTX}"
	rm ../Dockerfile
build:
	cp ./Dockerfile ../Dockerfile
	docker build -t "$(I_NAME)" -f ../Dockerfile "${BLD_CTX}"
	rm ../Dockerfile
run-cpu:
	docker run -it -p 8888:8888 -p 0.0.0.0:6006:6006 --name "$(C_NAME)_CPU" --memory="$(RAM_LIMIT)" -v "${HOST_SHARED_DIR}":/opt/shared -v "${HOST_SHARED_DIR_2}":/opt/shared_data "$(I_NAME)"
run:
	NV_GPU="$(GPU)" nvidia-docker run -it -p 8888:8888 -p 0.0.0.0:6006:6006 --name "$(C_NAME)" --memory="$(RAM_LIMIT)" -v "${HOST_SHARED_DIR}":/opt/shared -v "${HOST_SHARED_DIR_2}":/opt/shared_data "$(I_NAME)"
exec:
	docker exec -it "$(C_NAME)" bash
default_arguments:
	echo "GPU: $(GPU), IMAGE NAME: $(I_NAME) CONTAINER NAME: $(C_NAME), HOST PORT: $(HOST_PORT), LIMIT RAM TO: $(RAM_LIMIT)"
