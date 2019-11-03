import os
path = '/home/zh/11/2019'
allfile = os.listdir(path)
for f in allfile:
    #调试代码的方法：关键地方打上print语句，判断这一步是不是执行成功
    print(f)
    name = os.path.split(f)[1]
    if name in f and f.endswith(".txt.png"):
        print("原来的文件名字是:{}".format(f))
        #找到老的文件所在的位置
        old_file=os.path.join(path,f)
        print("old_file is {}".format(old_file))
        #指定新文件的位置，如果没有使用这个方法，则新文件名生成在本项目的目录中
        newname = name.split('.')[0] + '.png'
        new_file=os.path.join(path,newname)
        print("File will be renamed as:{}".format(new_file))
        os.rename(old_file,new_file)
        print("修改后的文件名是:{}".format(f))
