# Periodic Quantum Network for Time Series Forecasting
## Periodicity Modeling
Below is the run command for the first task periodic modeling, plus you can modify the simulated periodic data and hyperparameters via generate_periodic_data.py.
```shell
cd Periodicity_Modeling
bash ./run.sh
```
## Timeseries Forecasting
The following are the commands to run the timing forecasting task. First you need to prepare the common dataset for this task.

You can obtain data from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). 

```shell
cd Timeseries_Forecasting
bash scripts/Weather_script/QreTS.sh 
```
Also you can change the model by modifying the MODEL in the script file.

## Symbolic Formula Representation
Here is the symbolic formula indicating the run command for the task. The data generation code is run first, and then the script file for the corresponding model is run.

```shell
cd Symbolic_Formula_Representation
python gen_dataset.py
bash run_train_PQN.sh
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/thuml/Autoformer

https://github.com/YihongDong/FAN

https://github.com/romilbert/samformer
