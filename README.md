# GNNVis

run mnist example:

```
python -m experiments.train --batch-size=2 --k=20 --dsize=10000 --psize=200 --dataset=mnist
```

project file tree
```
 📦GNNVis
 ┣ 📂data
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜mnist.py
 ┃ ┗ 📜utils.py
 ┣ 📂experiments
 ┃ ┗ 📜train.py
 ┣ 📂gnnvis
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜gat.py
 ┃ ┣ 📜gnnvis.py
 ┃ ┣ 📜loss.py
 ┃ ┗ 📜sampler.py
 ┣ 📂knn
 ┃ ┣ 📂ANNOY
 ┃ ┃ ┣ 📜annoylib.h
 ┃ ┃ ┗ 📜kissrandom.h
 ┃ ┣ 📜Makefile
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜knn.cpp
 ┃ ┣ 📜knn.h
 ┃ ┣ 📜knn.py
 ┃ ┣ 📜knn_module.cpp
 ┃ ┣ 📜knnmodule.cpp
 ┃ ┣ 📜setup.py
 ┃ ┗ 📜sparse.py
 ┣ 📂utils
 ┃ ┣ 📜eval.py
 ┃ ┣ 📜metrics.py
 ┃ ┗ 📜plot.py
 ┗ 📜README.md
```

install networkx 2.3, since metis dependent it.
```
conda install -y networkx=2.3
```

install third part tools [metis](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz)

```
tar -zxvf metis-5.1.0.tar.gz && cd metis-5.1.0
make config shared=1 prefix=~/.local/
make install
export METIS_DLL=~/.local/lib/libmetis.so (on linux) or export
METIS_DLL=~/.local/lib/libmetis.dylib (on macos)
`pip install metis`
```

install coranking

```
# set up clang compiler in macos
export CC=clang
export CXX=clang++

pip install git+https://github.com/samueljackson92/coranking.git
```

