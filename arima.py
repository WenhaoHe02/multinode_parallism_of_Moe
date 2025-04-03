# 平稳性检测
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('data.csv')

# 将时间列转换为日期时间格式
data['date'] = pd.to_datetime(data['date'])

# 将日期列设置为索引
data.set_index('date', inplace=True)

# 检查数据是否平稳

# 平稳性检测
result = adfuller(data['value'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 如果 p 值小于 0.05，则拒绝原假设，认为数据是平稳的
if result[1] < 0.05:
    print("数据是平稳的")
else:
    print("数据是非平稳的")

# 如果数据不平稳，则进行差分处理
diff_data = data.diff().dropna()

# 再次进行平稳性检测
result = adfuller(diff_data['value'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 如果 p 值小于 0.05，则拒绝原假设，认为数据是平稳的
if result[1] < 0.05:
    print("数据是平稳的")
else:
    print("数据是非平稳的")

# 如果数据平稳，则进行 ARIMA 模型拟合
from statsmodels.tsa.arima.model import ARIMA

# 拟合 ARIMA 模型
model = ARIMA(data['value'], order=(1, 1, 1))
model_fit = model.fit()

# 打印模型摘要
print(model_fit.summary())

# 预测未来 10 个时间点的值
forecast = model_fit.forecast(steps=10)

# 打印预测结果
print(forecast)
# 保存预测结果到 Excel 文件
forecast_df = pd.DataFrame(forecast, columns=['value'])
forecast_df.to_excel('forecast.xlsx', index=False)
# 保存模型到文件
model_fit.save('arima_model.pkl')

# 加载模型
loaded_model = ARIMA.load('arima_model.pkl')

# 使用加载的模型进行预测
forecast = loaded_model.forecast(steps=10)

# 打印预测结果
print(forecast)

# 保存预测结果到 Excel 文件
forecast_df = pd.DataFrame(forecast, columns=['value'])