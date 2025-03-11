CUDA_IDX=3
DATASET='gbmlgg' # gbmlgg brca

mode='coattn' # 'coattn' 'pibd'
fusion='concat' # 'None' 'concat'

model_type='mcat'
g_model_type='ldvae'

FEATURE='CTransPath' # 'RN50-B' 'CTransPath'
wsi_encoding_dim=768
DATA_DIR=/home/zjj/zjj/data/TCGA/${DATASET}/ExpData/feats-l1-s256-${FEATURE}

alphas=(0.05 0.1 0.2 0.5 1.0)
betas=(0.05 0.1 0.2 0.5 1.0)

for alpha in ${alphas[@]}; do
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
        --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
        --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
        --g_condition --generator --warm_epoch 8 --max_epochs 30 \
        --alpha $alpha --beta 1.0 --decoder_mode specific
done

for beta in ${betas[@]}; do
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
        --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
        --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
        --g_condition --generator --warm_epoch 8 --max_epochs 30 \
        --alpha 0.1 --beta $beta --decoder_mode specific
done

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
    --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
    --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
    --g_condition --generator --warm_epoch 8 --max_epochs 30 \
    --alpha 0.1 --beta 1.0 --decoder_mode shared