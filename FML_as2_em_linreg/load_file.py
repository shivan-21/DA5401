#  load the data from A2Q1.csv
import numpy as np
import matplotlib.pyplot as plt
import os  



script_dir = os.path.abspath(os.path.dirname(__file__))
# Join that path with the filename
file_path = os.path.join(script_dir, 'A2Q1.csv')

# --- 3. Load data using the new, absolute path ---
data = np.loadtxt(file_path, delimiter=',')
plt.hist(data.flatten(), bins= 4, edgecolor='black')
plt.show()

print('Reached Here')