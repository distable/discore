import sys

session = 'play4time'
remote = False
dev = True
install = True

append_list = [
    'play4time',
    '--dev',
    '--run',
    # '--install',
    # '--vaic'
]

sys.argv.extend(append_list)

print(' '.join(sys.argv[1:]))
import discore