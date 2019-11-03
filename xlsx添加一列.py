from openpyxl import load_workbook

wb = load_workbook('/home/zh/123/151.xlsx')
ws = wb.worksheets[0]
ws.delete_cols(1,3) # 删除1,2,3列
ws.insert_cols(5)   # 第5列之前插入一列
for i,file in enumerate(file_path):
       for j ,row in enumerate(ws.rows):
           if j==0:
               row[4].value ='标题'
           elif j == i+1:
               row[4].value = i
wb.save('/home/zh/123/123.xlsx')
