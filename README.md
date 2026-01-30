# Pose Estimation

## Setup

Please note that this script runs faster on GPUs and would detect it if it's available.

### Conda Environment

For easier setup, you can use conda environment and run the following command after installing conda/miniconda.
```sh
conda env create -f environment.yml
conda activate pose-estimation
```

It will use python3.13.

### Python Environment

To setup using python run the following command on Linux/MacOS.
```sh
python3 -m venv .venv && source .venv/bin/activate
```

For Windows:
```sh
python -m venv .venv
.venv\Scripts\activate
```

#### Installing Dependencies

If you decided to use python environment, you would have to install the required packages.

After activating your environment, install the required packages:
```sh
pip install -r requirements.txt
```

## Run the Script

The script is a command line interface which can be invoked using the following command:

```sh
python run.py --input video.mp4 --output out.mp4 --json out.json
```

NOTE: if you're on windows, you might need to use python3 instead of python.

* The input flag specifies the input video file to be processed.
* The output flag specifies where to save the annotated video.
* The json flag specifies where to save the estimated poses for every frame as per requirements.

## Code Structure

The repo used MVC pattern, Models, Views and Controllers.

Models is where to keep model objects for data types, Controllers contains the business logic, views contains the command line interface logic.

Folders structure as follows:

```sh
├── configs # Configuration folder.
│   ├── config.py # Configuration file.
│   ├── ml_conf.py # YOLO model configuration for optimization and accuracy control.
│   ├── __init__.py
├── controllers # Controllers folder where all the heavy lifting is done.
│   ├── __init__.py
│   ├── video_processing.py # Video processing logic.
│   └── yolo.py # Extracting pos estimation logic.
├── environment.yml # Environment file for conda.
├── models # Models folder.
│   ├── __init__.py
│   └── video_models.py # Videos annotated data objects in pydantic.
├── README.md # Documentation file.
├── requirements.txt # Environment requirements file.
├── run.py # Entery point, used with command line.
└── views # Views file which contain the command line logic.
    └── cli.py
```

## Further Optimization


The current setup takes an 0.17 second average on 12th Gen Intel(R) Core(TM) i7-12700H for each frame. It also takes an average of 0.025 second on GPU NVIDIA GeForce RTX 3050.

Please note that I used in the `configs/config.py` configuration file model yolo26  `medium` which is already fast and optimized with further optimization in the `configs/ml_conf.py` which is optimized for one person object detection reduced image size to `640 pixel` for an faster processing time which both affects accuracy as well. 

I defaulted to `640` image size for safer choice, noting that according to my experiment, a `320` image size was sufficient in both accuracy and model performance with a 0.02 second per frame.

I already optimized this setup by caping frames to 30 frame per second if the processed video includes more than 30 frames per second.

Another possibility for further optimization is to use less accurate model with smaller and faster capabilities.

Another possible optimization is to make this machine learning model task specific, prune the model then fine tune again.

