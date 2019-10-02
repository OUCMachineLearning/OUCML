# USAGE - ./launch-docker-gpu.sh {abs-path-to-GDL-code}
docker run --rm --runtime=nvidia --network=host -it -v $1:/GDL gdl-image
