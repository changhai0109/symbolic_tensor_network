git clone https://github.com/astra-sim/stage
cd stage

# create a venv
python -m venv create .venv
source .venv/bin/activate
python -m pip install -r ./requirements.txt

# install chakra, optional for STAGE but if you want to use chakra tools it is prefered
cd ../astra-sim/extern/graph_frontend/chakra
# fix protobuf to use 6.* to fix conflicts to astra-sim docker
sed -i 's/protobuf==5\.\*/protobuf==6.\*/' pyproject.toml
python -m pip install .


