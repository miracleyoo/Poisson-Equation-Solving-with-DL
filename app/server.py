# server.js （server一直启动， 挂在服务器上）
from flask import Flask
from flask import request
app = Flask(__name__)
#
# import torch
# import os
# from torch.autograd import Variable
# from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
#
#
# class Config(object):
#     def __init__(self):
#         self.USE_CUDA            = torch.cuda.is_available()
#         self.NET_SAVE_PATH       = "./source/trained_net/"
#         self.MODEL               = 'MiracleWeightWideNet'
#         self.NUM_CHANNEL         = 2
#         self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-1-1_LESS_LAYER_TRAIN_ALL'
#         self.LINER_HID_SIZE      = 1024
#         self.LENGTH              = 41
#         self.WIDTH               = 9
#         self.NUM_CLASSES         = 369
#
#
# def dl_init():
#     opt = Config()
#     if opt.MODEL == 'MiracleWeightWideNet':
#         net = miracle_weight_wide_net.MiracleWeightWideNet(opt)
#     elif opt.MODEL == 'MiracleWideNet':
#         net = miracle_wide_net.MiracleWideNet(opt)
#     elif opt.MODEL == 'MiracleNet':
#         net = miracle_net.MiracleNet(opt)
#     elif opt.MODEL == 'MiracleLineConvNet':
#         net = miracle_lineconv_net.MiracleLineConvNet(opt)
#
#     NET_SAVE_PREFIX = opt.NET_SAVE_PATH + opt.MODEL + '_' + opt.PROCESS_ID + '/'
#     temp_model_name = NET_SAVE_PREFIX + "best_model.dat"
#     if os.path.exists(temp_model_name):
#         net, *_ = net.load(temp_model_name)
#         print("Load existing model: %s" % temp_model_name)
#         if opt.USE_CUDA:
#             net.cuda()
#             print("==> Using CUDA.")
#     else:
#         FileNotFoundError()
#     return opt, net
#
#
# def dl_solver(model_input, net, opt):
#     net.eval()
#     if opt.USE_CUDA:
#         inputs = Variable(torch.Tensor(model_input).cuda())
#         outputs = net(inputs)
#         outputs.cpu()
#     else:
#         inputs = Variable(torch.Tensor(model_input))
#         outputs = net(inputs)
#
#     outputs = outputs.data.numpy()
#     return outputs
#

@app.route('/')  # 一个get方法 可以看服务器是否启动
def hello_world():
    return 'Hello World!'


@app.route('/get-output', methods=['POST']) #GET requests will be blocked
def get_output():
    upload_file = request.files['image01'] # 上传的文件，以文件的形式上传
    print(request.form)  # 上传的数据{表单形式}
    input_data = request.form # 得到input
    # output = output_function(input_data) # 得到output的过程
    # generate(output)   #处理完数据之后看你要把它直接返回，还是生成一个文件
    return 'Todo...'


if __name__ == '__main__':
    app.run()

