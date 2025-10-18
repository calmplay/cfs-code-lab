
ROOT_PATH=""
DATA_PATH=""

GAN_ARCH
LOSS_TYPE
NITERS


nohup python -u main.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --GAN_arch $GAN_ARCH \
    --loss_type_gan $LOSS_TYPE \
    --niters_gan $NITERS \
    > output_${GAN_ARCH}_${LOSS_TYPE}_${NITERS}.log \
    2>&1 &