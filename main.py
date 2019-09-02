from dataset import *
from model import MyModel
import pandas as pd
daset = Dataset()
mymode = MyModel()
mymode.train(daset,batch_size=4,epoch=5)
test_names,test_labels = mymode.get_test_names_labels(daset)
sub = pd.DataFrame({"file": test_names,
                    "species": test_labels})
sub.to_csv("./submission.csv", index=False, header=True)
print('output file saved')