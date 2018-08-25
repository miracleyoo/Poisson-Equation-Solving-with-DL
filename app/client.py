# client.js （每次想要操作的时候就本地执行这个，会向服务器请求数据）

import requests

files = {'image01': open('01.jpg', 'rb')}
user_info = {'name': 'letian', 'data': [1, 2, 3, 4.0]}
output = requests.post("http://127.0.0.1:5000/get-output", data=user_info, files=files)
# 测试用上面这个就行
# 服务器上改成对应的url和端口

print(output.text)
