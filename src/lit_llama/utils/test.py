n1_str= '11\n22'
n2_str= '33\n44'
print(n1_str)
print(repr(n1_str)[1:-1])
print(repr(n1_str)[1:-1] + repr(n2_str)[1:-1])
print(repr(n1_str)[1:-1] + "\n" + repr(n2_str)[1:-1])