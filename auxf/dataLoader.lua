function getImageData(small)

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

    train_images = torch.load(filePath .. 'CNN Model/' .. trainImageDataFile)
    test_images = torch.load(filePath .. 'CNN Model/' .. testImageDataFile)

    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
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

function getTextData()

    local train_texts = torch.load(filePath .. 'mirTagTr.t7')
    local test_texts = torch.load(filePath .. 'mirTagTe.t7')

    local train_labels_text = torch.load(filePath .. 'mirflickrLabelTr.t7') -- load from t7 file
    local test_labels_text = torch.load(filePath .. 'mirflickrLabelTe.t7') -- load from t7 file

    return train_texts.T_tr, test_texts.T_te, train_labels_text.L_tr, test_labels_text.L_te
end

function getData() 
end
