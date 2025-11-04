from os import listdir

res = [f"igcc/'{x}'" for x in listdir('.') if '.py' in x]
print(', '.join(res))