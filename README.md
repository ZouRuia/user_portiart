# hx_user_profile_inference

## predict_user_attribute
此文件夹为模型文件夹

## 使用方法
### 填写并生成config.ini
### ```python grpc_app.py`````

## 一些命令

### Python
```Python

--python_out=. : 编译生成处理 protobuf 相关的代码的路径, 这里生成到当前目录
--grpc_python_out=. : 编译生成处理 grpc 相关的代码的路径, 这里生成到当前目录
--proto_path=  : proto 文件的路径, 这里的 proto 文件在当前目录

python -m grpc_tools.protoc --python_out=grpc_file/ --grpc_python_out=grpc_file/ --proto_path=grpc_proto/ ./grpc_proto/side_feature.proto

```

