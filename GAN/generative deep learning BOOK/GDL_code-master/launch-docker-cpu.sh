# USAGE - ./launch-docker-cpu.sh {abs-path-to-GDL-code}
# - eg. to run from current directory:
#     ./launch-docker-cpu.sh $(pwd)
if [[ "$OSTYPE" == "darwin"* ]]; then
    docker run --rm -p 8888:8888 -it -v $1:/GDL gdl-image-cpu
else
    docker run --rm --network=host -it -v $1:/GDL gdl-image-cpu
fi
