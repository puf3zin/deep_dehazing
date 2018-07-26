# Single Image Dehazing Using Deep Learning

This repository contains the deep learning models used in the [DeepDive: An End-to-End Dehazing Method Using Deep Learning](https://ieeexplore.ieee.org/abstract/document/8097344/) paper.

The datasets used to train this models are:

* [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Reside](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

### Citation

If you use this model in your research, please cite:

    @inproceedings{goncalves2017deepdive,
        title={DeepDive: An End-to-End Dehazing Method Using Deep Learning},
        author={Goncalves, Lucas Teixeira and Gaya, Joel De Oliveira and Drews, Paulo and Botelho, Silvia Silva Da Costa},
        booktitle={Graphics, Patterns and Images (SIBGRAPI), 2017 30th SIBGRAPI Conference on},
        pages={436--441},
        year={2017},
        organization={IEEE}
    }

### Prerequisites

To run this model, you will need:

* Tensorflow
* Python Imaging Library (PIL)
* Numpy


### Running the model

To run this model, simply run the **main.py** python code.

Arguments:

* -h, --help: View the help message and exit
* -m, --mode MODE: Specify one of the possible modes:
    * train
    * evaluate
    * restore
    * dataset_manage
* **-a, --architecture ARCHITECTURE: Specify the architecture used in the model**
* **-d, --dataset DATASET: Specify the dataset implementation used to train the model**
* **-l, --loss LOSS: Specify the loss implementation used to train the model**
* **-o, --optimizer OPTIMIZER: Specify the optimizer implementation used to train the model**
* -g, --dataset_manager DATASET_MANAGER
* -e, --evaluate EVALUATE
* --evaluate_path EVALUATE_PATH
* -p, --execution_path EXECUTION_PATH

The items highlighted in **bold** are obligatory for all modes except *dataset_manage*.