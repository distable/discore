# --------------------------------------------------------------------------------
# This is a file intended to be the default configuration for stable-core.
# Important: user_conf.py must import * from this file
#            you may copy the line below
#
# from src_core.conf import *
# --------------------------------------------------------------------------------


# Do not remove these, we are making them available by default in user_conf.py
from src_core.classes.Munchr import Munchr
import random
from src_core.classes.paths import short_pid

# Server
# ------------------------------------------------------------
ip = '0.0.0.0'
port = 5000
share = False

print_timing = False
print_more = False

def choice(values):
    def func(*args, **kwargs):
        return random.choice(values)

    return func
