git clone https://github.com/astra-sim/astra-sim.git
cd astra-sim
git submodule update --init --recursive

docker build -t astrasim .
cd ..

docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -v "$PWD":/app \
  -w /app \
  astrasim \
  bash
