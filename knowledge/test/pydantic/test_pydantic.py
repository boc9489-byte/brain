from pydantic import BaseModel, Field, ValidationError

class Student(BaseModel):
    name:str = Field(...,description = "姓名")
    age:int = Field(...,description = "年龄")
    score:float = Field(default_factory=float,description = "成绩")

s1 = Student(name="张三", age=20, score=95.4)
s2 = Student(name="张三", age=20)

print(s1) # name='张三' age=20 score=95.4

print(s2) # name='张三' age=20 score=0.0

print(type(s1.model_dump()))  # <class 'dict'>

print(type(s1.model_dump_json())) # <class 'str'>  {"name":"张三","age":20,"score":95.4}

try:
    s3 = Student(name="王武"
    # ,age="110岁"
    )
    print(s3)
except Exception as e:
    print(str(e))

s4 = Student(name="赵六", age=13, hobby = "篮球")
print(s4)  # name='赵六' age=13 score=0.0

data = {"name":"张三","age":20,"score":95.4}
s6=Student(**data)
print(s6) # name='张三' age=20 score=95.4


