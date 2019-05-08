bs1 = random.randint(1, 9)
bs2 = random.randint(0, 2)
batchSize = random.randint(1, 2000)
l1Nodes = random.randint(40, 1000)
l1Reg1 = random.randint(0, 9)
l1Reg2 = random.randint(-5, -1)
l1Reg = l1Reg1 * 10 ** l1Reg2
l1Dropout = random.randint(0, 80) / 100
l2Nodes = random.randint(4, 1000)
l2Dropout = random.randint(0, 80) / 100
l2Reg1 = random.randint(0, 9)
l2Reg2 = random.randint(-5, -1)
l2Reg = l2Reg1 * 10 ** l2Reg2
lr1 = random.randint(1, 9)
lr2 = random.randint(-6, -1)
lr = lr1 * 10 ** lr2

oldDf = pd.read_csv('C:\\Users\\chris\\Google Drive\\Python\\exparams.csv')

columnNames = ['Batch Size', 'L1 Nodes', 'L1 Regularization', 'L1 Dropout', 'L2 Nodes', 'L2 Regularization',
               'L2 Dropout', 'Learning Rate', 'Accuracy']

modelSpecs = pd.DataFrame([batchSize, l1Nodes, l1Reg, l1Dropout, l2Nodes, l2Reg, l2Dropout, lr,
                           modelSaver.tempBest]).T
modelSpecs.columns = columnNames
oldDf = oldDf.append(modelSpecs, ignore_index=True, sort=False)


oldDf.to_csv('C:\\Users\\chris\\Google Drive\\Python\\hyperparams.csv', index=False)
