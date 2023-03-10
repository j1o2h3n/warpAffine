# warpAffine()
TensorFlowJS implementation of affine transformation function warpAffine().


## Explain
基于tfjs实现3D-tensor图片的仿射变换操作函数warpAffine()实现。

```javascript
warpAffine(src, matrix, dsize=[112,112],borderValue=[0,0,0],bilinear_interpolation=true)

:param src: tensor, 输入图片张量, shape[H, W, C]
:param matrix: Tuple, 仿射矩阵. shape[2, 3]
:param dsize: Tuple, shape[W, H]. 输出的size
:param borderValue: Tuple, 空白处填充值,[0,0,0] or [255,255,255] etc.
:param bilinear_interpolation: bool, 是否选择双线性插值
:return: tensor, shape[dsize[1], dsize[0], C]
```


## Requirements
- tfjs
- mathjs


## Replace
本demo仅仅单纯使用tfjs手撕实现仿射变换操作而已，其亦可通过tfjs库自带的tf.image.transform()函数实现相同效果。


## References
- https://blog.csdn.net/weixin_42398658/article/details/121019668
- https://blog.csdn.net/qq_40939814/article/details/117966835

