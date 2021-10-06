from configparser import ConfigParser

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