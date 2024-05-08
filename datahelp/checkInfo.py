"""
读取一个ECXCEL文件，并且判断如果BEHAVIOR_PARAM列包含字符串"IF_"就把这一行转存到另一个excel文件，读取完成后需要把转存的excel保存的当前文件夹
"""
import pandas as pd
import json

import pandas as pd

def excel_to_csv(excel_file_path, csv_file_path):
    # 使用pandas读取Excel文件
    df = pd.read_excel(excel_file_path)
    
    # 将读取的数据保存为CSV文件
    df.to_csv(csv_file_path, index=False)


"""
读取文档内的投融资数据
"""
def readTrz():
    # 读取 Excel 文件，假设文件名为 'input.xlsx'
    input_file_path = '数据记录.xlsx'
    df = pd.read_excel(input_file_path)

    # 判断 'title' 列是否包含字符串 'IF_'
    filtered_df = df[df['BEHAVIOR_PARAM'].str.contains('IF_', na=False)]
    # 保存筛选后的数据到一个新文件 'output.xlsx'
    output_file_path = '投融资操作记录.xlsx'
    filtered_df.to_excel(output_file_path, index=False)

"""
1、筛选掉行内包含："filterSos":""的
2、对于filterCycx：进行处理
"""
def process_behavior_param(param,zdpz):
    # 检查是否包含"filterSos":""
    if '"filterSos":""' in param:
        return None  # 返回 None 代表忽略这行

    # 检查是否包含"filterSos"
    if 'filterSos' in param:
        try:
            # 解析 JSON 字符串
            data = json.loads(param)
            # 提取需要的数据
            mview = data.get('mview', '')
            tableName = data.get('tableName', '')
            filterCycx = data.get('filterSos', '[]')
            # 解析 filterCycx JSON 字符串
            filterCycx = json.loads(filterCycx)
            mvname=""
            filter_data=[]
            for f in filterCycx:
                field = f['field']
                for index, row in zdpz.iterrows():
                    # 检查FIELD和MVIEW值是否与另外两个变量相等
                    if row['FIELD'] == field and row['MVIEW'] == mview:
                        mvname=row["TIPCONTENT"]
                        break
            
            # 提取 filterCycx 中的 field, filter, value
            filter_data = [
                {'field': f['field'],'value': f['value'],'title':f['title'],'type':f['type']}
                for f in filterCycx if f['value'].strip()  # 过滤掉 value 为空的项
            ]
            if(len(filter_data)==0):
                return None  
            # filter_data = [{'field': f['field'],'value': f['value'],'title':f['title'],'type':f['type']} for f in filterCycx]
            # 重新组装 JSON 数据
            result_json = json.dumps({
                'mview': mview,
                'mvname':mvname,
                'tableName': tableName,
                'filter': filter_data
            }, ensure_ascii=False)
            return result_json
        except json.JSONDecodeError:
            return None 
        
    if 'filterCycx' in param:
        try:
            # 解析 JSON 字符串
            data = json.loads(param)
            # 提取需要的数据
            mview = data.get('mview', '')
            tableName = data.get('tableName', '')
            filterCycx = data.get('filterCycx', '[]')
            # 解析 filterCycx JSON 字符串
            filterCycx = json.loads(filterCycx)
            # 提取 filterCycx 中的 field, filter, value
            # filter_data = [{'field': f['field'],'value': f['value'],'title':'','type':''} for f in filterCycx]
            mvname=""
            filter_data=[]
            for f in filterCycx:
                if f['value'].strip():
                    continue
                field = f['field']
                filename=""
                for index, row in zdpz.iterrows():
                    # 检查FIELD和MVIEW值是否与另外两个变量相等
                    if row['FIELD'] == field and row['MVIEW'] == mview:
                        mvname=row["TIPCONTENT"]
                        filename=row["TZMC"]
                        break
                filter_data.append({'field': f['field'],'value': f['value'],'title':filename,'type':'1'})
            # filter_data = [
            #     {'field': f['field'],'value': f['value'],'title':filename,'type':'1'}
            #     for f in filterCycx if f['value'].strip()  # 过滤掉 value 为空的项
            # ]
            if(len(filter_data)==0):
                return None  
            # 重新组装 JSON 数据
            result_json = json.dumps({
                'mview': mview,
                'mvname':mvname,
                'tableName': tableName,
                'filter': filter_data
            }, ensure_ascii=False)
            return result_json
        except json.JSONDecodeError:
            return None 

def checkData():
    zdpz = pd.read_excel('字段配置.xlsx')
    # 读取 Excel 文件
    df = pd.read_excel('投融资操作记录.xlsx')
    # 对 BEHAVIOR_PARAM 进行处理并过滤数据
    df['Processed_PARAM'] = df['BEHAVIOR_PARAM'].apply(process_behavior_param,zdpz=zdpz)
    df_filtered = df.dropna(subset=['Processed_PARAM'])  # 删除 Processed_PARAM 为 None 的行
    # 选择需要的列
    df_output = df_filtered[['ID', 'BEHAVIOR_TYPE', 'Processed_PARAM']]

    # 保存到新的 Excel 文件
    df_output.to_excel('投融资操作记录_mark.xlsx', index=False)


# 主函数入口
# Python 文件直接被运行时（而不是作为模块被导入时），执行下面缩进的代码块。
# 这种写法常用于将一个 Python 文件既作为可执行脚本运行，又可以作为模块被其他 Python 文件导入和调用。
# 在这种写法下，可以将一些需要在脚本直接运行时执行的初始化逻辑或主要功能封装在 main() 函数中，
# 并通过 if __name__ == "__main__": 条件来控制它们的执行。
if __name__ == "__main__":
    excel_to_csv(excel_file_path='aspcoder_train.xlsx', csv_file_path='aspcoder_train.csv')