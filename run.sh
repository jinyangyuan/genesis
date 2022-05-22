#!/bin/bash

function run_model {
    run_name=$name'_'$model
    data_config='datasets/customized_config.py'
    model_config='models/'$model'_config.py'
    path_data=$folder_base'/'$name'.h5'
    python $run_file \
        --run_name $run_name \
        --data_config $data_config \
        --model_config $model_config \
        --path_data $path_data \
        --img_size $img_size \
        --K_steps $K_steps \
        --K_steps_general $K_steps_general
}

export PYTHONPATH=$PYTHONPATH:./forge

run_file='main.py'
folder_base='../compositional-scene-representation-datasets'

for model in 'genesis' 'genesisv2'; do
    name='mnist'
    img_size=64
    K_steps=6
    K_steps_general=8
    run_model
    name='dsprites'
    img_size=64
    K_steps=7
    K_steps_general=10
    run_model
    name='clevr'
    img_size=128
    K_steps=8
    K_steps_general=12
    run_model
    name='shop'
    img_size=128
    K_steps=8
    K_steps_general=12
    run_model
done
