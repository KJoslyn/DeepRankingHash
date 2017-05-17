function getImageAndTextDataNuswide()

    -- Tags for text data are also included

    -- local file = '/home/kjoslyn/datasets/nuswide/trainQueryValSplitNormalized.mat'
    -- local split = matio.load(file, 'split')
    -- torch.save('trainQueryValSplitNormalized.t7', split)

    local split = torch.load('/home/kjoslyn/datasets/nuswide/trainQueryValSplitNormalizedLarge.t7')
    return split.trainSet, split.querySet, split.valSet -- QuerySet will act as test set
end

function getImageAndTextDataMirflickr()

    -- Tags for text data are also included
    local split = torch.load('/home/kjoslyn/datasets/mirflickr/trainQueryValSplitUnnormalized.t7')

    -- Mirflickr data is unnormalized
    local mean = {} -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = split.trainSet.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        split.trainSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        split.valSet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        split.querySet.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = split.trainSet.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        split.trainSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        split.valSet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        split.querySet.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    return split.trainSet, split.querySet, split.valSet
end

function normalizeNuswideImageData()
    -- This function is only to be used once to normalize the nuswide data

    -- split = matio.load('blah.mat') -- This is a mat file with uint8 for the unnormalized data

    split.trainSet.data = split.trainSet.data:transpose(3,4)
    split.trainSet.data = split.trainSet.data:transpose(2,3)
    split.valSet.data = split.valSet.data:transpose(3,4)
    split.valSet.data = split.valSet.data:transpose(2,3)
    split.querySet.data = split.querySet.data:transpose(3,4)
    split.querySet.data = split.querySet.data:transpose(2,3)

    split.trainSet.data = split.trainSet.data:float()
    split.valSet.data = split.valSet.data:float()
    split.querySet.data = split.querySet.data:float()

    if not matio then
        matio = require 'matio'
    end

    avgStd = matio.load('/home/kjoslyn/datasets/nuswide/avgStdevTrainSetLarge.mat')
    avgStd = avgStd.avgStd
    for i = 1,3 do
        split.trainSet.data[{ {}, {i}, {}, {} }]:add(-avgStd.means[i][1])
        split.querySet.data[{ {}, {i}, {}, {} }]:add(-avgStd.means[i][1])
        split.valSet.data[{ {}, {i}, {}, {} }]:add(-avgStd.means[i][1])

        split.trainSet.data[{ {}, {i}, {}, {} }]:div(avgStd.stdev[i][1])
        split.querySet.data[{ {}, {i}, {}, {} }]:div(avgStd.stdev[i][1])
        split.valSet.data[{ {}, {i}, {}, {} }]:div(avgStd.stdev[i][1])
    end
end

function getTextDataNuswide() 

    if not matio then
        matio = require 'matio'
    end

    local split = matio.load('/home/kjoslyn/datasets/nuswide/trainQueryValSplitTextOnlyLarge.mat')
    split = split.split;
    return split.trainSet, split.querySet, split.valSet -- QuerySet will act as test set
end

function getTextDataMirflickr() 

    local split = torch.load('/home/kjoslyn/datasets/mirflickr/trainQueryValSplitTextOnly.t7')
    return split.trainSet, split.querySet, split.valSet -- QuerySet will act as test set
end

-- ///////////////////////
-- OLD METHODS
-- ///////////////////////

function getImageDataMirflickrOld(small)

    print('Getting image data')

    local trainImageDataFile = nil
    local testImageDataFile = nil
    if small then 
        print('**** Warning: Loading small datasets')
        trainImageDataFile = 'mirflickr_trainset_small.t7'
        testImageDataFile = 'mirflickr_testset_small.t7'
    else
        trainImageDataFile = 'mirflickr_trainset.t7'
        testImageDataFile = 'mirflickr_testset.t7'
    end

    local train_images = torch.load(g.filePath .. 'CNN Model/' .. trainImageDataFile)
    local test_images = torch.load(g.filePath .. 'CNN Model/' .. testImageDataFile)

    -- Mirflickr data is unnormalized
    local mean = {} -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = train_images.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        train_images.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        test_images.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = train_images.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        train_images.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        test_images.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    print('Done getting image data')

    return train_images, test_images
end

function getTextDataMirflickrOld()

    local train_texts = torch.load(g.filePath .. 'mirTagTr.t7')
    local test_texts = torch.load(g.filePath .. 'mirTagTe.t7')

    local train_labels_text = torch.load(g.filePath .. 'mirflickrLabelTr.t7') -- load from t7 file
    local test_labels_text = torch.load(g.filePath .. 'mirflickrLabelTe.t7') -- load from t7 file

    return train_texts.T_tr, test_texts.T_te, train_labels_text.L_tr, test_labels_text.L_te
end
