import xlwt, xlrd

def read_excel(path, sheet_idx=0, sheet_name=None):
    file = xlrd.open_workbook(path)
    try:
        sheet = file.sheet_by_index(sheet_idx) if sheet_name == None else file.sheet_by_names(sheet_name)
        return sheet
    except:
        print('Open sheet fail')


def get_itea_item(sheet, row_list=None, col_list=None):
    '''
        Input: row_list: 行号(y), col_list: 列号(x)
        Return: Itea, 以行为单位迭代地返回值
    '''
    if row_list == None:
        row_list = list(range(0, sheet.nrows))
    if col_list == None:
        col_list = list(range(0, sheet.ncols))

    for row in row_list:
        content = []
        for col in col_list:
            content.append(sheet.cell_value(row, col))
        yield content

def get_total_item(sheet, row_list=None, col_list=None):
    '''
        Input: row_list: 行号(y), col_list: 列号(x)
        Return: 直接完整的返回表中特定的值[[XXXX, XXXX, XXXX], [], []]
    '''
    if row_list == None:
        row_list = list(range(0, sheet.nrows))
    if col_list == None:
        col_list = list(range(0, sheet.ncols))
    
    total_content = []
    for row in row_list:
        content = []
        for col in col_list:
            content.append(sheet.cell_value(row, col))
        total_content.append(content)
    return total_content

def create_excel(sheet_name='XZY'):
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet(sheet_name)
    return workbook, worksheet

def write_excel(path, content):
    '''
        content: list[[A, B, C, D], [],..., []]
        后缀名: xls
    '''
    book, sheet = create_excel()
    for row in range(len(content)):
        for col in range(len(content[row])):
            sheet.write(row, col, label = str(content[row][col]))
    book.save(path)
    

if __name__ == '__main__':
    sheet = read_excel('E:\\dataset\\毕设数据\\label\\CDL_ColorMap.xlsx')
    # aa = get_itea_item(sheet)
    # for item in aa:
    #     print(item)
    bb = get_total_item(sheet)
    print(bb)
