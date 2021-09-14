# pytorch-image-classification

## TODO

- [ ] Read imagenette
- [ ] Train a ResNet18 on imagenette

## Debug


在CPU上用gloo跑

```bash
python train.py -d --dist-backend gloo -c configs/000.jsonnet
```