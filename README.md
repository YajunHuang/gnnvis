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
 ┃ ┣ 📜fmnist.py
 ┃ ┗ 📜utils.py
 ┣ 📂experiments
 ┃ ┗ 📜train.py
 ┣ 📂gnnvis
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜gat.py
 ┃ ┣ 📜gnnvis.py
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜predict.py
 ┃ ┗ 📜sampler.py
 ┣ 📂knn
 ┃ ┣ 📂ANNOY
 ┃ ┃ ┣ 📜annoylib.h
 ┃ ┃ ┗ 📜kissrandom.h
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜knn.cpp
 ┃ ┣ 📜knn.h
 ┃ ┣ 📜knn.py
 ┃ ┣ 📜knnmodule.cpp
 ┃ ┣ 📜setup.py
 ┃ ┗ 📜sparse.py
 ┣ 📂utils
 ┃ ┣ 📜eval.py
 ┃ ┣ 📜metrics.py
 ┃ ┗ 📜plot.py
 ┗ 📜README.md
```

install knn module

```
cd knn
python setup.py install (or python setup.py build_ext --inplace)
```

install coranking

```
# set up clang compiler in macos
export CC=clang
export CXX=clang++

pip install git+https://github.com/samueljackson92/coranking.git
```

Running in GPU

```
pip install dgl-cuXXX==0.4.3
```

