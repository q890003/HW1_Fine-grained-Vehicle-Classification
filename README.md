---
Selected-Topics-in-Visual-Recognition-using-Deep-Learning HW1
---
# Fine-grained Vehicle-Classification
[Reproducing the work](#Reproducing the work)  
[Training](#Training)  
[Inference](#Inference)  

## Reproducing the work
- Enviroment Installation
    1. install annoconda
    2. create python3.x version 
        ```
        take python3.6 for example
        $ conda create --name (your_env_name) python=3.6
        $ conda activate (your_env_name)
        ```
    3. install pytorch ([check GPU version](https://www.nvidia.com/Download/index.aspx?lang=cn%20))
        - [pytorch](https://pytorch.org/get-started/locally/)
- Project installation
1. Download Official Image
    ```
    $ pip install kaggle
    $ cd ~/.kaggle
    $ homepage www.kaggle.com -> Your Account -> Create New API token
    $ mv ~/Downloads/kaggle.json ./
    $ chmod 600 ./kaggle.json
    $ kaggle competitions download -c cs-t0828-2020-hw1
    ```
2. clone this repository
    ``` 
    git clone https://github.com/q890003/HW1_Fine-grained-Vehicle-Classification.git
    ```
3. put **data/**, create **checkpoints/**, **results/** under the root dir of this project. 
    ```
    |- HW1_Fine-grained-Vehicle-Classification
        |- data/
            |- training_data/
                |- 000001.jpg
                |- 000002.jpg
                ...
            |- testing_data/
                |- 000001.jpg
                |- 000002.jpg
                ...
            |- training_labels.csv
        |- checkpoints/
        |- results/
        |- datasets/
        |- models/
        |- optimizers/ranger/
        |- transforms/
        |- .README.md
        |- train.py
        |- eval.py
        ...
    ```
4. [Downoad fine-tuned parameters](https://drive.google.com/file/d/1Q5SbN6o7zoV5DUDaGPaBPQTEN1qTWbel/view?usp=sharing)
    
## Training

```
$ python train.py
``` 
## Inference

```
$ python eval.py
```

