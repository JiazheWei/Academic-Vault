
### torch.mean()
用法：
```python
a.torch.randn(1,2,3,4)
a=torch.abs(a).mean(a,dim=[1,2,3].keepdim=True)
```

mean操作可简单认为对哪个维度取均值就将哪个维度压缩为1，`keepdim`声明要不要将压缩的维度保留。


