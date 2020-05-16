import numpy as np
import os 
import glob
import h5py
import json

from Helper.ImitationLearning import image_size

MIN_BATCH_SIZE = 32
MAX_BATCH_SIZE = 256
batchSize= MIN_BATCH_SIZE
MAX_SPEED = 25.0

class Loader():
    def __init__(self, path, name , branches , branchCommands):
        self.path = path
        self.name = name
        self.branches = branches
        self.branchCommands = branchCommands
        self.dict = {'Path': self.path}
        self.load_dict()
    def files(self):
        return glob.glob(self.path + '*.h5')
    def get_branches(self):
        return self.branches
    def get_branchCommands(self):
        return self.branchCommands
    def create_branches(self):
        for i, branch in enumerate(self.branches):
            self.dict[branch] = {'Commands':self.branchCommands[i], 
                                'Files':{},
                                'Count':0,
                                }
        
    def reshape_branches(self):
        for branch in self.branches:
            files = list(self.dict[branch]['Files'].keys())
            fileIndex =np.random.randint(0 , len(files))
            while self.dict[branch]['Count'] >= self.min_count:
                #i = np.random.randint(0 , len(self.dict[branch]['Files'][files[fileIndex]] )) 
                self.dict[branch]['Files'][files[fileIndex]].pop(0)
                if len(self.dict[branch]['Files'][files[fileIndex]]) == 0: 
                    del self.dict[branch]['Files'][files[fileIndex]]
                    files = list(self.dict[branch]['Files'].keys())
                    fileIndex =np.random.randint(0 , len(files))    
                self.dict[branch]['Count'] -=1
            self.write_json()
            
    def write_json(self):
        with open(self.name+'.json', 'w') as fp:
            fp.write(json.dumps(self.dict,  indent=2))
        fp.close()
        
    def load_dict(self):
        contents = os.listdir(os.getcwd())
        if self.name+'.json' in contents:
            try:
                with open(self.name+'.json', 'r') as fp:
                    self.dict = json.load(fp)
                fp.close()
            except:
                print("Wrong Json deleting it and generating new")
                os.remove(self.name+'.json')
                self.load_dict()

            if self.dict['Path'] != self.path:
                print("Wrong Json deleting it and generating new")
                os.remove(self.name+'.json')
                self.load_dict()
        else:
            self.dict = {} # Empty dictionary as aprecautionary measure
            self.create_branches()
            self.dict['Path'] = self.path
            print(f"generating new json from {len(self.files())} files")
            for file in self.files():
                try:
                    data = h5py.File(file , 'r')
                    for branch in self.branches:
                        for index in range(data['rgb'].shape[0]):
                            if data['targets'][index][24] in self.dict[branch]['Commands']:
                                if file not in self.dict[branch]['Files'].keys():
                                    self.dict[branch]['Files'][file] = [index]
                                    self.dict[branch]['Count'] += 1
                                else:
                                    self.dict[branch]['Files'][file].append(index)
                                    self.dict[branch]['Count'] += 1
                except: 
                    print(file)
            for branch in self.branches:
                print(f"Branch: {branch} DataPoints : {self.dict[branch]['Count']}")
            self.min_count = min(self.dict[branch]['Count'] for branch in self.branches)
            self.reshape_branches()


def genBranch(branch,command, batchSize):
    while True:  # to make sure we never reach the end
        batchX = np.zeros((batchSize, image_size[0], image_size[1], image_size[2]))
        batchY = np.zeros((batchSize, 28))
        branchOneHot = {0:0 ,1:0, 2: 0 , 3:1 , 4:2, 5:3}
        counter = 0
        while counter <= batchSize - 1:
            try:
                files = list(branch['Files'].keys())
                fileIndex =np.random.randint(0 , len(files))
                data = h5py.File(files[fileIndex], 'r')
                Indexes = branch['Files'][files[fileIndex]]
                dataIdx = np.random.randint(0, len(Indexes))
                while dataIdx<len(Indexes): #200 images
                    batchX[counter] = data['rgb'][Indexes[dataIdx]]#np.concatenate([data['rgb'][Indexes[dataIdx]] ,data['depth'][Indexes[dataIdx]]], 1)
                    targets = data['targets'][Indexes[dataIdx]]
                    #print(command)  
                    if command == "Speed":
                        targets[24] = 4
                    elif command == "Intent":
                        targets[24] = 5 
                    else:
                        targets[24] = branchOneHot[int(targets[24])]  
                    targets[10] /= MAX_SPEED
                    batchY[counter] = targets
                    counter+= 1
                    dataIdx+=1
                    if counter >= batchSize:
                        break    
                data.close()
            except:
                print(fileIndex , dataIdx)
        yield (batchX, batchY) 