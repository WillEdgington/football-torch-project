from .preprocess import tensorDatasetsFromMatchDf, createDataLoaders, prepareData

if __name__=="__main__":
    tensorDict = tensorDatasetsFromMatchDf() # create and save datasets
    loaderDict = prepareData() # load dataloaders