import re

with open('index.html') as f:
    text = f.read()

# Replace document.getElementById with $
# define const $ = id => document.getElementById(id); at the start of script
text = text.replace("<script>", "<script>\nconst $ = id => document.getElementById(id);")
text = text.replace("document.getElementById", "$")

with open('index.html', 'w') as f:
    f.write(text)

print('New length:', len(text))
