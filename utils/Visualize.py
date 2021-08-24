import numpy as np

#visualize
def visualize(labels,outputs,name,outlier = 0):
  color = np.random.randint(244, size=(100, 3))
  for i in range(len(labels)):
    label = int(labels[i])
    if(outlier == 1 and label == 0):
      outputs[i,3:] = [0,0,0]
    else:
      outputs[i,3:] = color[label]
  txt_name = "./result/result_" + name + ".txt"
  np.savetxt(txt_name, outputs, fmt='%20.5f', delimiter= ';')  