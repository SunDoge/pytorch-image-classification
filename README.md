# pytorch-image-classification

## Install

初始化submodule
```bash
git submodule init
git submodule update
```

更新submodule
```bash
git submodule foreach git pull
```

## TODO

- [ ] Read imagenette
- [ ] Train a ResNet18 on imagenette

## Debug


在CPU上用gloo跑

```bash
python train.py -d --dist-backend gloo -c configs/000.jsonnet
```