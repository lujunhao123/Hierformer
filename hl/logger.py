import os
import logging
from datetime import datetime

def result_print(result, info_name='Evaluate',folder_path='./result/result_tem/',model_name="1"):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
    print("========== {} {} results ==========".format(info_name,model_name))
    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[16], total_MAE[20]))
    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[4] * 100, total_MAPE[8] * 100, total_MAPE[12] * 100, total_MAPE[16] * 100, total_MAPE[20] * 100))
    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[4], total_RMSE[8], total_RMSE[12], total_RMSE[16], total_RMSE[20]))
    print("---------------------------------------")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, 'result.txt')
    f = open(file_path, 'a')
    f.write("\n")
    f.write("========== {} {} results ==========".format(info_name,model_name))
    f.write(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_MAE[0], total_MAE[4], total_MAE[8], total_MAE[12], total_MAE[16], total_MAE[20]))
    f.write("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_MAPE[0] * 100, total_MAPE[4] * 100, total_MAPE[8] * 100, total_MAPE[12] * 100, total_MAPE[16] * 100,
    total_MAPE[20] * 100))
    f.write("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f\n" % (
    total_RMSE[0], total_RMSE[4], total_RMSE[8], total_RMSE[12], total_RMSE[16], total_RMSE[20]))
    f.write("---------------------------------------\n")