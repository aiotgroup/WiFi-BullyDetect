
# 汉字占两英文宽，则每存在一个汉字少填充一个长度
def pad_len(string, length):
    return length - len(string.encode('GBK')) + len(string)

def log_f_ch(*str_list, str_len: list = None):

    if str_len is None:
        str_len = [10 for _ in range(len(str_list))]
    else:
        assert len(str_len) == len(str_list) ,"length of 'str_len' must equal to length of 'str_list'"

    output_str = ''
    for i, str in enumerate(str_list):
        output_str += "{0:<{len1}}\t".format(str, len1=pad_len(str, str_len[i]))
    return output_str
