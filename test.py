import pandas as pd
from data_processor import *
from bayesian_regression import *
from performance_evaluation import *

price_1 = pd.read_csv('.//price4.csv')
price_2 = pd.read_csv('.//price5.csv')
price_3 = pd.read_csv('.//price6.csv')
n = [180, 360, 720]
price_reshaped_1 = price_1.values.reshape((1,  -1))
price_reshaped_2 = price_2.values.reshape((1, -1))
price_reshaped_3 = price_3.values.reshape((1, -1))
data_pro = ProcessData(price_reshaped_1, n, 100, 20)

ts = data_pro.generate_time_series()
cluster = data_pro.find_clusters()
effective = data_pro.select_effective_clusters()

test_model = Prediction(effective, price_reshaped_2, n)
p = test_model.predict_delta_p()
print(p)

bench = np.random.randn(100, 1)
eval = Evaluation(price_reshaped_3, 720, p, 1, 0.01, bench, 100)
delta = eval.visual_account()
#risk = eval.calculate_max_drawdown()
#print(risk)
#eval.calc_mean_return_by_quantile()


