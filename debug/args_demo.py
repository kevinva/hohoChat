import argparse

parser = argparse.ArgumentParser(description = "used for test")

parser.add_argument('--version', '-v', action = 'version', version = '%(prog)s 1.0', help = 'show the version of this program')
parser.add_argument('--debug', '-d', action = 'store_true', help = 'debug mode')

# parser.add_argument('name')       # 必选参数
# parser.add_argument('age', type = int)     # 必选参数
parser.add_argument('--gender', '-g', default = 'male', choices = ['male', 'female'])  # 可选参数
parser.add_argument('--name', '-n')

args = parser.parse_args()

# print(f"{args.name}")
# print(f"{args.age}")
print(f'{args.gender}')
print(f'{args.name}')

print('=== end ===')
