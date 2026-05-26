# class CUtils(object):
#     def __init__(self):
#         pass

#     def create_var(self, var_name, value):
#         if not var_name in self.__dict__:
#             self.__dict__[var_name] = value

class CUtils:
    def create_var(self, var_name: str, value) -> bool:


        if not isinstance(var_name, str) or not var_name.isidentifier():
            raise ValueError(f"Invalid property name:{var_name}")
    
        if value is None:
            raise ValueError("Attribute value cannot be None")
        # 3. 安全新增属性
        if not hasattr(self, var_name):
            setattr(self, var_name, value)
            return True
        return False