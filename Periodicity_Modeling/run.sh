GPU=0
export CUDA_VISIBLE_DEVICES=${GPU}

periodicType=d2_s1
modelName=$1
path=./${periodicType}_${modelName}
python3 -u ./test.py \
--model_name ${modelName} \
--periodic_type ${periodicType} \
--path ${path}

wait $!