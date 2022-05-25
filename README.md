# skforget

Personal implementation of different machine learning algorithms from scratch without any dependencies

## Quick Start

### Linear Regression

```python
import skforget as skf

X, Y = skf.make_regression()

reg = skf.LinearRegression()
reg.fit(X, Y)
Y_pred = reg.predict(X)
```

![Linear Regression](./bin/linear_regression.png)
