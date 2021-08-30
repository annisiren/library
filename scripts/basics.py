
# Save object to file
import pickle
with open(file_name, 'wb') as f:
    pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)

# Open file
with open(file_name, 'w') as f: # 'w', 'r', 'wb', 'a'
    f.readline()


# Doesn't use regular codex
import codecs
with codecs.open(file_name, 'w', encoding='utf-8') as f:
    f.write(text)


import time

start_time = time.time()
print("My program took", time.time() - start_time, "to run")
