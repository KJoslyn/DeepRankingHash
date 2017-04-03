function subtractMean(data)

    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

        stdv[i] = data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
        data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    return data
end

function getImageData()

    print('Getting image data')

    train_images = torch.load(filePath .. 'CNN Model/mirflickr_trainset.t7')
    test_images = torch.load(filePath .. 'CNN Model/mirflickr_testset.t7')
    train_images.data = subtractMean(train_images.data)
    test_images.data = subtractMean(test_images.data)
    trainset[I] = train_images.data
    testset[I] = test_images.data

    train_labels_image = train_images.label
    test_labels_image = test_images.label

    print('Done getting image data')
end

function getTextData()

    local train_texts = torch.load(filePath .. 'mirTagTr.t7')
    local test_texts = torch.load(filePath .. 'mirTagTe.t7')
    trainset[X] = train_texts.T_tr
    testset[X] = test_texts.T_te

    train_labels_text = torch.load(filePath .. 'mirflickrLabelTr.t7') -- load from t7 file
    train_labels_text = train_labels_text.L_tr
    test_labels_text = torch.load(filePath .. 'mirflickrLabelTe.t7') -- load from t7 file
    test_labels_text = test_labels_text.L_te
end
