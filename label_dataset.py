from json import loads, dump

goto0 = '\x1B[0;0H'
green = '\x1B[1;32m'
bold = '\x1B[1m'
reset = '\x1B[0m'
clearall = '\x1B[2J'

in_file = open('input.json')

json = []

for line in in_file:
    json.append(loads(line))

in_file.close()

i = 0

need = []
give = []
useless = []
category = []


def write():
    out_file = open("classified.json", "w")

    dump({
        "need": need,
        "give": give,
        "useless": useless,
    }, out_file, indent=2)

    out_file.close()


input_text = '(first letter) back/need/give/useless: '

print(clearall, goto0, end='')

while i < len(json):
    tweet = json[i]["tweet"]
    print(
        bold, green, f'({i}/{len(json)}) ? ', reset, bold, tweet, reset, sep=''
    )

    choice = input(input_text)
    if choice == 'b':
        if i > 0:
            [need, give, useless][category.pop()].pop()
            i -= 1
        print(clearall, goto0, end='')
        continue
    if choice == 'n':
        need.append(json[i])
        category.append(0)
    elif choice == 'g':
        give.append(json[i])
        category.append(1)
    elif choice == 'u':
        useless.append(json[i])
        category.append(2)
    else:
        print(clearall, goto0, sep='')
        continue

    i += 1
    print(clearall, goto0, end='')
    write()

write()
