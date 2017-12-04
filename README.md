steps to start the server

Step 1: run the docker-gabriel/run_shell.sh file
	(You many need root permission [you can do: sudo su])
	
Step 2: in a new terminal, ssh into the docker container by running the follwoing command:  ./docker_login.sh gab
(You many need root permission [you can do: sudo su])
	Note: here "gab" is the name of the docker running instance that was created in step1 



	Step 2a: Post ssh, navigate to the following directory:
			cd gabriel/server/bin/
	Step 2b: Run gabriel-ucomm by:
			./gabriel-ucomm

Step 3: in a new terminal, ssh into the docker container by running the follwoing command:  ./docker_login.sh gab
(You many need root permission [you can do: sudo su])
	Step 3b: Post ssh, navigate to the following directory:
			 cd /gabriel/server/bin/example-proxies

	Step 3c: run the text detection engine by:
			python gabriel-proxy-text-detection.py


Other info: 

To stop the docker container:
	docker stop gab  # here gab is the name of the instance

If you have any issues with docker reusing the same image/container:
	docker system prune

docker_login.sh contents:
	docker exec -it $1 /bin/bash

run_shell.sh file contents: 
	docker run -p 8021:8021 -p 9090:9090 -p 9098:9098 -p 9099:9099 -p 9100:9100 -p 9111:9111 -p 10101:10101 -p 10102:10102 -p 10103:10103 -p 22222:22222 --name gab -it gab_temp