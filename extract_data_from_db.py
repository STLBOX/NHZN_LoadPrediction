from app import *
import pickle
# 从 data.db 中提取数据并保存为 np 方便训练模型
# total_power days*24  and weather days*24*weather_feature
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # power_save = []
    # st_day = datetime(2024, 5, 13, 0, 0, 0)
    # for i in range(200):
    #     power_day = Totalpower.query.filter_by(date=st_day.date()) \
    #         .with_entities(Totalpower.total_power) \
    #         .order_by(Totalpower.number) \
    #         .all()
    #
    #     weather_day = Weather.query.filter_by(date=st_day.date()) \
    #         .with_entities(Weather.date, Weather.hour, Weather.temp, Weather.humidity, Weather.wind_speed) \
    #         .order_by(Weather.hour) \
    #         .all()
    #
    #     power_sub = []
    #     if power_day and weather_day:
    #         power_sub = [[itemp[0], itemw[0].month, itemw[0].day, itemw[0].weekday(), itemw[1],
    #                       itemw[2], itemw[3], itemw[4]] for itemp, itemw in zip(power_day, weather_day)]
    #         if np.array(power_sub).shape == (24, 8):
    #             power_save.append(power_sub)
    #
    #     st_day += timedelta(days=1)
    # power_save = np.array(power_save)
    # print(f'保存数据大小 : {power_save.shape}')
    # np.save('data/power.npy', power_save)
    st_day = datetime(2024, 5, 12, 0, 0, 0)
    users = Power.query.filter(Power.date == st_day.date(), Power.power < 0) \
                 .with_entities(Power.user_id, Power.meter_id, Power.power).all()
    less_zero_user = []
    less_zero_power = []
    for user in users:
        if user[0] not in less_zero_user:
            less_zero_user.append(user[0])
            less_zero_power.append((user[1]))
    data = pd.DataFrame({"user_id": less_zero_user, "power_id": less_zero_power})
    data.to_excel("sihong_less_zeros_id.xlsx")









