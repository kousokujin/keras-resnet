import json
import os


def include_dict(dict,key,value):
    '''辞書型の配列で指定keyとvalueが存在するか
    '''

    keyvalue = dict.get(key)

    if keyvalue == None:
        return False
    else:
        if keyvalue == value:
            return True
        else:
            return False

'''
    fixed_datas = {"epoch":10,dataset:mnist}
    x1 = "batch_size"
    x2 = "method"
'''
def generate_2bar(json_path,fixed_datas,x1,x2,y):
    with open(json_path) as f:
        s = f.read()
        json_data = json.loads(s)

    datas = []
    for d in json_data["result"]:
        isMatched = True
        keys = fixed_datas.keys()
        for k in keys:
            value = fixed_datas[k]
            if include_dict(d,k,value) == False:
                isMatched = False
        if isMatched == True:
            datas.append(d)

    x1_values = []
    x2_values = []
    for d in datas:
        x1_values.append(d[x1])
        x2_values.append(d[x2])
    x1_values = set(x1_values)
    x2_values = set(x2_values)

    y_values = []
    for v2 in x2_values:
        y_temp = []
        for v1 in x1_values:
            for d in datas:
                if d[x1] == v1 and d[x2] == v2:
                    y_temp.append(d[y])
        y_values.append(y_temp)

    return (x1_values,x2_values,y_values)
        

def json_bar2_graph(graph_config,path):
    '''
    graph_config = {
        title,
        file_name,
        json_path,
        fixed_datas,
        x1,
        x2,
        y,
        auto
    }
    '''
    x1_val,x2_val,y_val = generate_2bar(graph_config["json_path"],graph_config["fixed_datas"],graph_config["x1"],graph_config["x2"],graph_config["y"])

    x1_val = [str(s) for s in x1_val]
    x2_val = [str(s) for s in x2_val]

    y_str = []
    for y1 in y_val:
        y_array = [str(s) for s in y1]
        y_str.append(y_array)

    table = {
        "title" : graph_config["title"],
        "file_name": graph_config["file_name"],
        "x_name" : graph_config["x1"],
        "y_name" : graph_config["y"],
        "x": x1_val,
        "y": y_str,
        "labels": x2_val,
        "auto": graph_config["auto"],
        "type": "bar"
    }
    
    with open(path,mode='w') as f:
        json.dump(table,f,indent=4, sort_keys=True, separators=(',',': '))


config = {
    "title": "test",
    "file_name": "test.png",
    "json_path" : "result/20190327_test.json",
    "fixed_datas": {"epoch": "2","dataset": "mnist"},
    "x1": "batch_size",
    "x2": "method",
    "y": "accuracy",
    "auto": "true"
}
#datas = {"epoch": "2","dataset": "mnist"}
#x1_val,x2_val,y_val = generate_2bar("result/20190327_test.json",datas,"batch_size","method","accuracy")
#json_bar2_graph(config,"test_table.json")
            