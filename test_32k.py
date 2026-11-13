# Let's see what happens if we use $ = id => document.getElementById(id)
with open('index.html') as f:
    text = f.read()

import re
# Count occurrences of document.getElementById
count = len(re.findall(r'document\.getElementById', text))
print('document.getElementById count:', count)
