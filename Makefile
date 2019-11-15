GPU = "0,1"  # split by ,
C_NAME = "speech_test_cont"
RAM_LIMIT = "24g"
I_NAME = "speech_test_img"

build-nocache:
	docker build --no-cache -t "$(I_NAME)" -f Dockerfile .
build:
	docker build -t "$(I_NAME)" -f Dockerfile .
run:
	NV_GPU="$(GPU)" nvidia-docker run -it -p 8888:8888 --name "$(C_NAME)" --memory="$(RAM_LIMIT)" --rm -v /home/rwerner/SpeechProcessing/shared:/opt/shared "$(I_NAME)"
exec:
	docker exec -it "$(C_NAME)" bash
default_arguments:
	echo "GPU: $(GPU), IMAGE NAME: $(I_NAME) CONTAINER NAME: $(C_NAME), HOST PORT: $(HOST_PORT), LIMIT RAM TO: $(RAM_LIMIT)"
