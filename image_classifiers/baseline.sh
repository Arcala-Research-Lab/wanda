
#Convex

python main.py --model convnext_base \
    --data_path /home/yichx14/datasets/ILSVRC/Data/CLS-LOC \
    --resume /home/yichx14/re/wanda/image_classifiers/model_weights/convnext/convnext_base_1k_224_ema.pth \
    --prune_metric wanda \
    --prune_granularity row \
    --sparsity 0.5 

* Acc@1 82.724 Acc@5 96.388 loss 0.704
Accuracy of the network on 50000 test images: 82.72400%

* Acc@1 82.696 Acc@5 96.368 loss 0.705
Accuracy of the network on 50000 test images: 82.69600%


#deit_base
python main.py --model convnext_base \
    --data_path /home/yichx14/datasets/ILSVRC/Data/CLS-LOC \
    --resume /home/yichx14/re/wanda/image_classifiers/model_weights/convnext/convnext_base_1k_224_ema.pth \
    --prune_metric wanda \
    --prune_granularity layer \
    --sparsity 0.5 \
    --layerwise_powers_json /home/yichx14/re/wanda/image_classifiers/two.json

