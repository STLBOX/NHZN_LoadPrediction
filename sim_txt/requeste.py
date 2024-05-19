import requests
import json
import pandas


log = open('log.txt', 'w', encoding='utf8')
with open('config.txt', 'r', encoding='utf8') as f:
    config_str = f.read()
config_dict = json.loads(config_str)

# 解析配置
'''
json.loads: str -> dict
json.dumps: dict -> str
'''
url0 = config_dict['url0']
headers0 = config_dict['headers0']
data0 = config_dict['data0']

url1 = config_dict['url1']
headers1 = config_dict['headers1']
data1 = config_dict['data1']
params1 = config_dict['params1']
json_body = json.dumps(params1)
log.write(json.dumps(url0)+'\n'+json.dumps(headers0)+'\n'+json.dumps(data0)+'\n'+json.dumps(url1)+'\n'
          +json.dumps(headers1)+'\n'+json.dumps(data1)+'\n'+json.dumps(params1))
log.write('\n'+'#####################result########################')
# post请求，获取token
try:
    response = requests.post(url=url0, headers=headers0, data=data0)
    log.write('\n'+response.text)
    # 解析响应
    dict_response = json.loads(response.content)
    # 获得
    access_token = dict_response['access_token']  # dict_response['content']['out']
    log.write('\n' + access_token)
    # 获得请求服务
    headers1['Authorization'] = access_token
except:
    log.write('\nrequests0 error!')

try:
    response = requests.post(url=url1, headers=headers1,  data=json_body)
    log.write('\n' + '#####################requests 2########################')
    log.write('\n' + response.content.decode())
    log.write('\n' + response.request.body)
    log.write('\n' + str(response.request.headers))
except:
    log.write('\nrequests1 error!')
log.close()
