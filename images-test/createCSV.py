import os


if __name__ == '__main__':
    csvFile = open("inputs.csv", "w")
    files = os.listdir("./")
    

    for fileName in files :
        if fileName.endswith(".jpg") :
            array_from_string = fileName.split('_')
            csvFile.write(fileName + "," + array_from_string[0] + "\n")