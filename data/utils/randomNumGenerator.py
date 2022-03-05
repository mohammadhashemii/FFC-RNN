from random import randint
import locale

total_number = 0


def thousands_seprator(num):
    '''

    :param num: an integet number
    :return: String format of the input number which is separated by ,(comma)
    '''

    locale.setlocale(locale.LC_ALL, 'en_US')
    str_num = locale.format_string("%d", num, grouping=True)
    return str_num


def random_n_base(n, end=1, step=1, start=1, count=16, mode='deafult'):
    temp = []

    if mode == "with_zeros":
        for k in range(4, 12):
            for c in range(count):
                digit_num = randint(k + 1, 12)  # number of digits
                range_start = 10 ** (digit_num - 1)
                range_end = 10 ** (digit_num) - 1
                rand_num = randint(range_start, range_end)
                valid_num = rand_num - (rand_num % 10 ** (k))
                # temp.append(digits.en_to_fa(thousands_seprator(valid_num)))
                temp.append(valid_num)
        return temp

    for i in range(start, end, step):
        range_start = i * 10 ** (n - 1)
        range_end = (i + 1) * (10 ** (n - 1)) - 1
        for c in range(count):
            rand_num = randint(range_start, range_end)
            valid_num = rand_num
            # temp.append(digits.en_to_fa(thousands_seprator(valid_num)))
            temp.append(valid_num)
    return temp


def generate_random_num():
    # list of total numbers
    tn = []

    # print("5-digit: 10,000 - 11,000 - ... - 19,000 ###############################################################")
    tn.append(sorted(random_n_base(4, 20, start=10, count=100)))
    # print(tn[-1]) #5 digits               10,000 - 11,000 - ... - 19,000

    # print("5-digit: 20,000 - 30,000 - ... - 90,000 ###############################################################")
    tn.append(sorted(random_n_base(4, 91, start=20, step=10, count=100)))
    # print(tn[-1])  # 5 digits    20,000 - 30,000 - ... - 90,000

    # print("6-digit: 100,000 - 200,000 - ... - 900,000 ###############################################################")
    tn.append(sorted(random_n_base(6, 10, count=100)))
    # print(tn[-1])             #6 digits               100,000 - 200,000 - ... - 900,000

    # print("7-digit: 1,000,000 - 2,000,000 - ... - 9,000,000 ###############################################################")
    tn.append(sorted(random_n_base(7, 10, count=100)))
    # print(tn[-1]) #7 digits               1,000,000 - 2,000,000 - ... - 9,000,000

    # print("8-digit: 10,000,000 - 11,000,000 - ... - 19,000,000 ###############################################################")
    tn.append(sorted(random_n_base(7, 20, start=10, count=100)))
    # print(tn[-1])             #8 digits               10,000,000 - 11,000,000 - ... - 19,000,000

    # print("8-digit: 20,000,000 - 30,000,000 - ... - 90,000,000 ###############################################################")
    tn.append(sorted(random_n_base(7, 91, start=20, step=10, count=100)))
    # print(tn[-1])  # 8 digits               20,000,000 - 30,000,000 - ... - 90,000,000

    # print("9-digit: 100,000,000 - 200,000,000 - ... - 900,000,000 ###############################################################")
    tn.append(sorted(random_n_base(9, 10, count=100)))
    # print(tn[-1])             #9 digits               100,000,000 - 200,000,000 - ... - 900,000,000

    # print("10-digit: 1,000,000,000 - 2,000,000,000 - ... - 9,000,000,000 ###############################################################")
    tn.append(sorted(random_n_base(10, 10, count=100)))
    # print(tn[-1])#10 digits              1,000,000,000 - 2,000,000,000 - ... - 9,000,000,000

    # print("11-digit: 10,000,000,000 - 11,000,000,000 - ... - 19,000,000,000 ###############################################################")
    tn.append(sorted(random_n_base(10, 20, start=10, count=100)))
    # print(tn[-1])            #11 digits              10,000,000,000 - 11,000,000,000 - ... - 19,000,000,000

    # print("11-digit: 20,000,000,000 - 30,000,000,000 - ... - 90,000,000,000 ###############################################################")
    tn.append(sorted(random_n_base(10, 91, start=20, step=10, count=100)))
    # print(tn[-1])  # 11 digits              20,000,000,000 - 30,000,000,000 - ... - 90,000,000,000

    # print("12-digit: 100,000,000,000 - 200,000,000,000 - ... - 900,000,000,000 ###############################################################")
    tn.append(sorted(random_n_base(12, 10, count=100)))
    # print(tn[-1])            #12 digits              100,000,000,000 - 200,000,000,000 - ... - 900,000,000,000

    # print("with zeros:")
    tn.append(sorted(random_n_base(4, count=150, mode="with_zeros")))
    # print(tn[-1])

    output = []
    for singleType in tn:
        output.extend(singleType)

    # total number of types
    print("# of number types : " + str(len(tn)))

    # total number of samples
    total = 0
    for type in tn:
        total += len(type)
    print("# of total samples : " + str(total))
    print('Note: In order to change the number of samples, modify it in randomNumGenerator.py')
    return output

# if __name__ == "__main__":
#    numbers = generateRandomNum()
