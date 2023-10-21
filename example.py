from sklearn.datasets import load_boston
X, y = load_boston()

### Use MondrianForests for variance estimation
from skgarden import MondrianForestRegressor
mfr = MondrianForestRegressor()
mfr.fit(X, y)
y_mean, y_std = mfr.predict(X, return_std=True)

### Use QuantileForests for quantile estimation
from skgarden import RandomForestQuantileRegressor
rfqr = RandomForestQuantileRegressor(random_state=0)
rfqr.fit(X, y)
y_mean = rfqr.predict(X)
y_median = rfqr.predict(X, 50)

print(f"Score: {y_median}")
