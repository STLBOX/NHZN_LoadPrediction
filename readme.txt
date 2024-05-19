# 建立数据库 初始化
in the python shell：
from app import db
db.create_all()

# add
db.session.add()
db.session.commit()

# query 调用 class方法


# 全局初始化数据 说明
weather_data
nums 标识时间(周期为15min) [0, 5, 9...] -> [0, 1, 2...]h
power_data


# 数据库功能说明

# 数据库初始化
init_db: 重置数据库
init_user() ->  电表id初始化，固定的表单，记录用户， 可能需要每年更新
init_power() -> 用户功率数据库初始化，全部采用默认值power_data初始化？  K-2  
init_weather() -> init_weather_partial()  # 初始化气象数据，request前2天的气象数据 ，K日数据采用K-1日代替 ； 查询是否存在，存在exit，不存在add  K-1,K-2,request   K初始化K-1   如果失败只能替换为默认值

# 数据库在线请求修改
get_token()
getSHID()
getPowerData()  # 获取用户功率，需要一个一个请求，指定日期，一一请求，请求前一天的数据，失败或者错误都保持前一天数值
getWeatherDataP()  # 请求间隔24h，每天7点30，获取天气预测数据  当天的所有的数据；查询是否存在，存在更新（当前时间之后更新）[8, 9, 10, 11...]，不存在add， 响应错误，使用历史同时期的值
if not request_error:  请求是否错误
	if existing_weather is None:  是否查询到
getWeatherDataN()  # 请求间隔1h，分钟数为5的时候请求，每隔1h保存实时状态；检查是否存在，存在更新，不存在添加

# 数据库删除多余数据
delete_hst_data()

# 数据库查询，供模型训练，以及显示
delete_hst_data()

# 算法输入，历史气象数据(k-1),历史功率数据(k-1),预测气象数据(k)
# 输出,预测工业部分总体功率数据(k)
predict_power()


# config.txt 说明
url0: token
url1: 请求功率数据
url2: 请求泗洪县气象中心的id  "Authorization":"", 需要token 
url3: 请求天气预报数据   "Authorization":"", 需要token    parma "id":"101191304", 需要id 
url4: 请求历史天气数据   "Authorization":"", 需要token    parma "id":"101191304", 需要id    timeout=5s
url5: 请求实时天气数据    "Authorization":"", 需要token       parma "ids":"101191304", 需要ids  timeout=3s


# flask 基本说明
css imag js 位置确定：href="{{ url_for('static', filename='css/style.css') }}"

# jinja2 常用的模版语法
{{ }}    用 {{ ... }} 包裹变量名
{% if user.is_authenticated %}    使用 Python 的表达式
{% endif %}

{% for item in items %}  循环语句

 {% endfor %}

{{ text|upper }} 过滤器

块：在base.html中添加  {% block content %}{% endblock %}
可以被子类继承并修改： 
{% extends 'base.html' %}  # 引入将被复用的模版base.html
{% block content %}  # 修改的位置为 content块
  <h1>Content Goes Here</h1>
{% endblock %}


#  数据动态更新,设计数据接口


导航： 时间  nowtime 和预测实时 nowpower 功率显示分辨率(1h)
nowpower  依赖 DATA['power']

工业功率监测表格：更新时间  每天7.30 请求前一天所有数据，更新曲线所有数据   update_power_table() 更新用户功率曲线，并获得预测结果   1 days
    "power_table_data": {
                        'usernums': "",
                        'pk1max': "",
                        'pk1min': "",
                        'pkmax': "",
                        'pkmin': "",
                        },

预测气象数据表格：更新时间 1h，每隔1h请求气象数据     1 hours
    "weather_table_data":{
                        'wtstate': '**',
                        'wtmaxtemp': 0,
                        'wtmintemp': 0,
                        'wtmaxhum': 0,
                        'wtminhum': 0,
                        'wtmaxws': 0,
                        'wtminws': 0
                        },
用户数据详情：更新时间 每天 7.30 显示前一天的最高瞬时功率以及出现时间

'user_top_table': {
'user_number': ['*', '*', '*', '*', '*'],
'user_id': ['*', '*', '*', '*', '*'],
'user_name': ['*', '*', '*', '*', '*'],
'user_cat': ['*', '*', '*', '*', '*'],
'max_power': [0, 0, 0, 0, 0],
'max_power_time': [0, 0, 0, 0, 0]}


预测功率曲线曲线：  已经保存在全局对象中，委托给工业功率监测表格时候更新数据

"power_curve": {
"ptimes":
"ptimeData": const
"power":
"pred_max":
"pred_min":
}


气象数据曲线： 也可以更新  (Weather.temp, Weather.humidity, Weather.wind_speed, Weather.prec）
"weather_curve":{
"times":
"timeData":
"windxData":
"windsData":
"temData"：
"rainData":
"humData":
}



// 程序说明
启动... 
初始化数据库数据： 
delete_user_table()       init_user() # 基本保持不变
init_power()  # 如果数据K-2日数据不存在，采用默认值power_data初始化
init_weather()  # 请求获得K-1, K-2数据   K采用K-1 初始化 ,
 请求--> 不存在数据初始化-->请求成功用请求结果，失败采用默认值weather_data

辅助请求的函数：
get_token()  
getSHID()

初始化数据均是获得历史数据，而接下来是获得建模十分重要的数据。
getPowerData()  # 如果没有K-1，则请求获得K-1数据，失败或者错误保持为K-2   getPowerData_threadingpool() 线程池实现版本
getWeatherDataP()  # 获得预测天气数据K  成功request新增或替换 不成功采用历史同期值
getWeatherDataN()  # 获得当前时刻的气象数据，并更新  天气的表格  #



// 下面这些更新的都应该很快，设置一个后台更新线程；指定时间去执行更新；
// 

update_nav();    // 间隔一个小时 **:05  更新一次
update_power_table();   // 7:20更新 --> 同时更新后台的全局对象 即用户的功率更新；并获得预测结果  DATA["power_curve"]   DATA["weather_curve"]  DATA['power_table_data']  DATA['weather_table_data']
update_weather_table();   // 间隔一个小时 **:05  更新一次  
update_user_top_table();  // 7:50更新  DATA['user_top_table']

update_weather_curve()   // 间隔一个小时 **:10  更新一次
update_power_curve()  // 7:45更新  -->  delete_hst_data()  执行一次删除




# 打包过程bug   flask路径问题   相对路径问题，原始接口不对
# 打包指定图标  pyinstaller -F --icon=static\img\favicon.ico app.py
# pip 镜像 -i https://pypi.tuna.tsinghua.edu.cn/simple app.py



# dm数据库设置
dm_config.txt 填写 数据库ip port 用户名 和 密码
dm_sql_query.txt 填写一条sql语句，查询结果显示在log中


