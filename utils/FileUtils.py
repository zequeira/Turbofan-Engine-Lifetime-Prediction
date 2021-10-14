from configparser import ConfigParser
from utils.DataManipulationUtils import *
from configConstants import *
import pandas as pd


class IniFilesReader:
    """
        Read ini files given filename
        Extract the root element by name from ini file
    """
    global fileObject

    def __init__(self, fileName):
        self.fileName = fileName
        self.__openFile(fileName)

    def __openFile(self,fileName : str):
        self.fileObject = ConfigParser()
        self.fileObject.read(fileName)
    
    def getProperty(self , name) -> object:
        return self.fileObject[name]



class EngineDataFiles:
    import os
    import re

    """
        Manipulate final engina data files
    """

    _iniFilesReader = IniFilesReader(FILES_INI_FILE_NAME)
    _iniRootFile=_iniFilesReader.getProperty(EngineSimulationConstants.ROOT)
    _iniPatternsRoot=_iniFilesReader.getProperty(EngineSimulationConstants.PATTERNS_ROOT)
    _TRAIN_FILES_FOLDER=_iniRootFile[EngineSimulationConstants.TRAIN_DATA_FOLDER]
    _FINAL_TRAIN_FILES_FOLDER=_iniRootFile[EngineSimulationConstants.TRAIN_DATA_OUTPUT_FOLDER]

    _TRAIN_FILE_NAME_PATTERN=_iniPatternsRoot[EngineSimulationConstants.TRAIN_FILES_NAME_PATTERN]

    def __init__(self):
        pass

    def readInitialTrainData(self,fileName : str) -> pd.DataFrame :
        """
            Read pretransformed data
        """
        names = ['engineNumber', 'cycleNumber','altitude', 'mach', 'TRA',
         'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
         'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
        file=self.os.path.join(self._TRAIN_FILES_FOLDER,fileName)

        if self.os.path.exists(file):
            trainDf : pd.DataFrame=pd.read_csv(file,header=None,delim_whitespace=" ",keep_default_na=False)
            trainDf.columns=names
        else:
            raise FileNotFoundError("File name : ",file)
        return trainDf

    def readFinalTrainData(self,fileName) -> pd.DataFrame:
        file=self.os.path.join(self._FINAL_TRAIN_FILES_FOLDER,fileName)
        if self.os.path.exists(file):
            df : pd.DataFrame=pd.read_csv(file,sep=" ",header=0,keep_default_na=False)
        else:
            raise FileNotFoundError("File name : ",file)
        return df
        
    def writeFinalTrainData(self):
        """
            Read every file from the the train folder given a file pattern and
            write a new file with all the extracted data
        """
        inputFilesPathRoot = self._TRAIN_FILES_FOLDER  
        outputDataFolder = self._iniRootFile[EngineSimulationConstants.TRAIN_DATA_OUTPUT_FOLDER]      
        outputFileName = self._iniRootFile[EngineSimulationConstants.OUTPUT_FILE_NAME]    

        
        folderFileNames = self.os.listdir(inputFilesPathRoot)
        print("Listed files in {} :\n {}".format(inputFilesPathRoot,folderFileNames))  
        matchedFiles=0
        for fileName in folderFileNames:
            if self.re.match(self._TRAIN_FILE_NAME_PATTERN,fileName):
                matchedFiles=matchedFiles+1
                #Read train file
                trainDf=self.readInitialTrainData(fileName)
                #Calculate RUL
                newDfWithRul=add_remaining_useful_life(trainDf)
                #Write the new registers in train file
                outputFile=self.os.path.join(outputDataFolder,outputFileName)
                if self.os.path.exists(outputFile):
                    print("Adding {} rows of data to {}".format(newDfWithRul.shape[0],outputFile))
                    newDfWithRul.to_csv(outputFile,mode='a',sep=" ",index=False)
                else:
                    print("Creating new file : ",outputFile)
                    newDfWithRul.to_csv(outputFile,mode='w',sep=" ",index=False)

        print("Matched files : {} , unmatched files : {}".format(matchedFiles,len(folderFileNames)-matchedFiles))