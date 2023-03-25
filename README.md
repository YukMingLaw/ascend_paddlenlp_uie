# ascend paddlenlp-uie
a program of running paddlenlp-uie on huawei-ascend platform

## requirement
pip:
```
paddlepaddle>=2.3.0
paddlenlp==2.4
paddle2onnx
onnxruntime
```
system:
```
ascend-toolkit >= 5.1.RC2
```

## get pdmodel and pdiparams
use the code to get 'uie-base' model as below
```
from paddlenlp import Taskflow
schema = ['Person', 'Organization']
ie = Taskflow('information_extraction', schema=schema, model="uie-base")
```
the model default download path is
```
/usr/local/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdiparams
/usr/local/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdmodel
```
copy the .pdiparams and .pdmodel to the path that you like
```
cp /usr/local/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdiparams {model_path}/model.pdiparams
cp /usr/local/.paddlenlp/taskflow/information_extraction/uie-base/static/inference.pdmodel {model_path}/model.pdmodel
```

## run uie on ascend
```
python3 infer_ascend.py --model_path_prefix {model_path}/model
```
get result be like:
```
1. Input text:
"北京市海淀区人民法院
民事判决书
(199x)建初字第xxx号
原告：张三。
委托代理人李四，北京市 A律师事务所律师。
被告：B公司，法定代表人王五，开发公司总经理。
委托代理人赵六，北京市 C律师事务所律师。"
2. Input schema:
[{'原告': ['出生日期', '委托代理人']}, {'被告': ['出生日期', '委托代理人']}]
3. Result:
{'原告': [{'end': 38,
         'probability': 0.9951229095458984,
         'relations': {'委托代理人': [{'end': 47,
                                  'probability': 0.74322509765625,
                                  'start': 45,
                                  'text': '李四'}]},
         'start': 36,
         'text': '张三'}],
 '被告': [{'end': 68,
         'probability': 0.8501815795898438,
         'relations': {'委托代理人': [{'end': 93,
                                  'probability': 0.7247400283813477,
                                  'start': 91,
                                  'text': '赵六'}]},
         'start': 65,
         'text': 'B公司'}]}
-----------------------------
```