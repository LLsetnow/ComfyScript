import random

chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
keys = [''.join(random.choice(chars) for _ in range(16)) for _ in range(10)]

print('"keys": {')
for i, k in enumerate(keys):
    if i < 9:
        print(f'    "{k}": {{"used": false, "used_by": null, "used_time": null}},')
    else:
        print(f'    "{k}": {{"used": false, "used_by": null, "used_time": null}}')
print('}')
