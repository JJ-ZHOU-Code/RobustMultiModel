CUDA_IDX=3
DATASET='gbmlgg' # gbmlgg brca

mode='coattn' # 'coattn' 'pibd'
fusion='concat' # 'None' 'concat'

model_type='mcat'
g_model_type='ldvae'

FEATURE='CTransPath' # 'RN50-B' 'CTransPath'
wsi_encoding_dim=768
DATA_DIR=/home/zjj/zjj/data/TCGA/${DATASET}/ExpData/feats-l1-s256-${FEATURE}

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
    --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
    --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
    --g_condition --generator --warm_epoch 5 --max_epochs 30

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
    --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
    --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
    --g_condition --generator --warm_epoch 8 --max_epochs 30

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
    --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
    --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
    --g_condition --generator --warm_epoch 15 --max_epochs 30

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python main.py \
    --split_dir tcga_${DATASET} --data_root_dir $DATA_DIR --feature_extractor ${FEATURE} \
    --wsi_encoding_dim $wsi_encoding_dim --fusion ${fusion} --mode ${mode} --model_type ${model_type} --g_model_type ${g_model_type} \
    --g_condition --generator --warm_epoch 20 --max_epochs 30
