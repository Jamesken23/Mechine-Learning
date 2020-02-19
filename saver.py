import os
import pandas as pd
from openpyxl.workbook import Workbook

# def save_to_excel(self,dataFrame,dataType,saveDir='resDir/'):
#     if not os.path.exists(saveDir):
#         os.makedirs(saveDir)
#     savePath=os.path.join(saveDir,'%s_on_semeval_all_%s.xlsx'%(self.name,dataType))
#     if len(sys.argv)==3:
#         savePath=os.path.join(saveDir,'%s_on_semeval_%s_%s_%s.xlsx'%(self.name,sys.argv[1],sys.argv[2],dataType))
#     if len(sys.argv)==4:
#         savePath = os.path.join(saveDir,'%s_on_semeval_%s_%s_%s_(%s).xlsx' % (self.name, sys.argv[1], sys.argv[2], dataType,sys.argv[3]))
#     dataFrame.to_excel(savePath)

# def save_to_excel(self,saveDir='resDir/',**kwargs):
#     if not os.path.exists(saveDir):
#         os.makedirs(saveDir)
#     savePath=os.path.join(saveDir,'%s.xlsx'%(self.name))
#     if len(sys.argv)==4:
#         savePath = os.path.join(saveDir,'%s_%s.xlsx' % (self.name,sys.argv[3]))
#     excel_writer = pd.ExcelWriter(savePath)
#     if 'dataFrames' in kwargs and 'sheetNames' in kwargs:
#         for dataFrame, sheetName in zip(kwargs['dataFrames'], kwargs['sheetNames']):
#             dataFrame.to_excel(excel_writer, sheet_name=sheetName)
#     elif 'dataFrame' in kwargs and 'sheetName' in kwargs:
#         kwargs['dataFrame'].to_excel(excel_writer, sheet_name=kwargs['sheetName'])

def save_to_excel(data_frameX,
                  sheet_nameX='Sheet1',
                  save_dir='./',
                  excel_name='result.xlsx',
                  index=True,
                  header=True):
    """
    将（多个）DataFrame数据结构和对应要存的（多个）sheet存在指定路径下
    不管是单个或者多个DataFrame都阔以
    :param data_frameX: （多个）DataFrame数据结构
    :param sheet_nameX: （多个）sheet的名称
    :param save_dir: 表格存储目录
    :param excel_name: 表格名称
    :param index: 是否保存索引
    :param header: 是否保存列名，在表格就是header
    :return:
    """
    # saveDir = os.path.split(savePath)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_excel_path = os.path.join(save_dir, excel_name)
    # savePath=os.path.join(saveDir,'%s.xlsx'%(self.name))
    # if len(sys.argv)==4:
    #     savePath = os.path.join(saveDir,'%s_%s.xlsx' % (self.name,sys.argv[3]))
    excel_writer = pd.ExcelWriter(save_excel_path)  # 打开一个excel表格
    if isinstance(data_frameX, list) and isinstance(sheet_nameX, list):
        # 如果有多个DataFrame构成的list
        for dataFrame, sheetName in zip(data_frameX, sheet_nameX):
            dataFrame.to_excel(excel_writer,
                               sheet_name=sheetName,
                               index=index,
                               header=header)
    elif isinstance(data_frameX, pd.DataFrame) and isinstance(sheet_nameX, str):
        # 如果仅有一个DataFrame
        data_frameX.to_excel(excel_writer,
                             sheet_name=sheet_nameX,
                             index=index,
                             header=header)
    excel_writer.save()  # 不能忘了


if __name__ == '__main__':
    import numpy as np

    arr1 = np.arange(0, 15).reshape((3, 5))
    index = [0, 1, 2]
    columns = ['A', 'B', 'C', 'D', 'E']
    df1 = pd.DataFrame(arr1, index, columns)
    print(df1)
    save_to_excel(df1, index=False, header=False)
