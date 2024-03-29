# IceCube Graph Neural Network
> Graph Neural Network model created for Kaggle competition [IceCube - Neutrinos in Deep Ice](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice)
> 
> This project partially based on Graphet model

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Model Architecture](#model-architecture)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- The goal of this model is to predict a neutrino particle’s direction, based on data from the "IceCube" detector, which observes the cosmos from deep within the South Pole ice.
- This project is designed to achieve similar performance to Graphnet using less complicated architecture and to achieve better performance with similar complexity
- What is the purpose of your project?
- Why did you undertake it?
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.8
- PyTorch Geometric - version 2.0
- Tech 3 - version 3.0


## Features
List the ready features here:
- Awesome feature 1
- Awesome feature 2
- Awesome feature 3


## Model Architecture
![Layers](./Img/Layers.PNG)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup
What are the project requirements/dependencies? Where are they listed? A requirements.txt or a Pipfile.lock file perhaps? Where is it located?

Proceed to describe how to install / setup one's local environment / get started with the project.


## Usage
Model works with input sensor data in parquet files.

`write-your-code-here`


## Project Status
Project is: _in progress_


## Room for Improvement

Room for improvement:
- Change head layers to transformer architecture to improve accuracy
- Further optimize model

To do:
- Rewrite code for better readability
- Remove all network code from notebook and add more examples


## Acknowledgements
- This project was inspired by work [Graph Neural Networks for Low-Energy Event Classification & Reconstruction in IceCube](https://arxiv.org/abs/2209.03042)
- This project was based on [GraphNet](https://github.com/graphnet-team/graphnet).


## Contact
Created by jhowlett79@gmail.com - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
