from pydantic import BaseModel

from typing import Optional

class student(BaseModel):
    
    name:str = 'vishal'
    age:Optional[int] = None  

details = {'age': '19'}
 
student_obj = student(**details)

print(student_obj)
