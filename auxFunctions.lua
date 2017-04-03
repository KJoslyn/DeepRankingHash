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

function getEpochPerm(epoch)

    epoch_pos_perm = pos_perm[ {{ epoch*posExamplesPerEpoch + 1 , (epoch + 1)*posExamplesPerEpoch }} ]
    epoch_neg_perm = neg_perm[ {{ epoch*negExamplesPerEpoch + 1 , (epoch + 1)*negExamplesPerEpoch }} ]

    return epoch_pos_perm, epoch_neg_perm
end

function getBatchPosOrNeg(array, batchNum, perm, batchSize)

    startIndex = batchNum * batchSize + 1
    endIndex = (batchNum + 1) * batchSize

    return array:index(1, perm[ {{ startIndex, endIndex }} ])
end

function getBatch(batchNum, epoch_pos_perm, epoch_neg_perm)

    pos_batch = getBatchPosOrNeg(pos_pairs, batchNum, epoch_pos_perm, posExamplesPerBatch)
    neg_batch = getBatchPosOrNeg(neg_pairs, batchNum, epoch_neg_perm, negExamplesPerBatch)

    batch_idx = torch.cat(pos_batch, neg_batch, 1)
    -- batch_idx = neg_batch -- TODO: Remove (for debugging)

    batch = {}
    batch.data = {}
    batch.data[I] = trainset[I]:index(1, batch_idx:select(2,1):long()) -- TODO: Fix long conversion in root pos_pairs and neg_pairs
    batch.data[X] = trainset[X]:index(1, batch_idx:select(2,2):long())
    batch.data[I] = batch.data[I]:cuda()
    batch.data[X] = batch.data[X]:cuda()

    setmetatable(batch, 
        {__index = function(t, i) 
                        return {t.data[I][i], t.data[X][i]} 
                    end}
    );

    function batch:size() 
        return self.data[1]:size(1) 
    end

    return batch, batch_idx
end


function doGetHashCodes(hasherInput, modality)

    if modality == I then
        pred = imageHasher:forward(hasherInput[modality]:cuda())
    else
        pred = textHasher:forward(hasherInput[modality]:cuda())
    end

    reshapedInput = nn.Reshape(L,k):cuda():forward(pred)
    maxs, indices = torch.max(reshapedInput, 3)
    return indices
end

function printStats(trainBatch, similarity_target)

    imHash = getHashCodes(trainBatch.data, I)
    teHash = getHashCodes(trainBatch.data, X)

    s = similarity_target:size(1)

    for i = 4,8 do
        similarity = torch.eq(imHash, teHash):sum(2):ge(i)
        numCorrect = torch.eq(similarity, similarity_target):sum()
        print(string.format("Accuracy%d = %.2f", i, numCorrect*100 / s))
    end

    -- similarity = torch.eq(imHash, teHash):sum(2):eq(L)
    -- numCorrect = torch.eq(similarity, similarity_target):sum()
    -- accuracy = numCorrect / similarity:size(1)

    -- print("Accuracy: " .. accuracy * 100)
end

function getHashCodes(data, modality)

    if modality == X then
        return doGetHashCodes(data, modality)
    elseif modality == I then
        N = data[I]:size(1)
        if N < 1000 then
            return doGetHashCodes(data, I)
        else
            local batchSize = 128
            hashCodes = torch.CudaLongTensor(N, L, 1)
            local numBatches = torch.ceil(N / batchSize)
            for b = 0, numBatches - 1 do
                startIndex = b * batchSize + 1
                endIndex = math.min((b + 1) * batchSize, N)
                batch = {}
                batch[I] = data[I][{{ startIndex, endIndex }}]
                hashCodes[{{ startIndex, endIndex}}] = doGetHashCodes(batch, I)
            end
            return hashCodes    
        end
    else
        print("Error: unrecognized modality")
    end
end

function calcMAP(fromModality, toModality, trainBatch, batchIdx) -- TODO: Remove 3rd parameter

    K = 50

    -- queryCodes = getHashCodes(testset, fromModality)
    queryCodes = getHashCodes(trainBatch, fromModality) -- TODO: Switch back!!
    -- databaseCodes = getHashCodes(trainset, toModality) -- Currently the database is only the trainset
    databaseCodes = getHashCodes(trainBatch, toModality) -- TODO: Switch back!!!

    if fromModality == I then
        -- queryLabels = test_labels_image:float()
        -- databaseLabels = train_labels_text:float()
        queryLabels = train_labels_image:float():index(1, batchIdx:select(2,1):long())
        databaseLabels = train_labels_text:float():index(1, batchIdx:select(2,2):long())
    else
        -- queryLabels = test_labels_text:float()
        -- databaseLabels = train_labels_image:float()
        queryLabels = train_labels_text:float():index(1, batchIdx:select(2,1):long())
        databaseLabels = train_labels_image:float():index(1, batchIdx:select(2,2):long())
    end

    -- Q = 1
    Q = queryCodes:size(1)
    sumAPs = 0
    for q = 1,Q do
        query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1, 1)
        ne = torch.ne(query, databaseCodes):sum(2)
        ne = torch.reshape(ne, ne:size(1))
        res, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

        res2, ind2 = torch.sort(res)
        topkIndices = ind:index(1,ind2)

        qLabel = queryLabels[q]

        if trainOnOneBatch then
            -- TODO: This is for testing only
            numSimilar = 0
            D = databaseLabels:size(1)
            for d = 1,D do
                dLabel = databaseLabels[d]
                dotPod = torch.dot(qLabel, dLabel)
                if dotPod > 0 then
                    numSimilar = numSimilar + 1
                end
            end
            numSimilar = math.min(numSimilar, K)
        else
            numSimilar = K
        end

        AP = 0
        correct = 0
        for k = 1,K do

            kLabel = databaseLabels[topkIndices[k]]
            dotProd = torch.dot(qLabel, kLabel)
            if dotProd > 0 then
                correct = correct + 1
                AP = AP + (correct / k) -- add precision component
            end
        end
        AP = AP / numSimilar -- Recall component: same as multiplying each precision component by 1/K. This assumes >= K instances are similar
        sumAPs = sumAPs + AP
    end
    mAP = sumAPs / Q

    return mAP
end

function calcBatchClassAccuracy(classifier, trainBatch, modality, batchIdx)

    local testModel = nil

    roundedOutput = classifier:forward(trainBatch[modality]):round()
    if modality == X then
        labels = train_labels_text:index(1, batchIdx:select(2,1):long()):cuda()
    else
        labels = train_labels_image:index(1, batchIdx:select(2,2):long()):cuda()
    end

    numInstances = trainBatch[1]:size(1)
    dotProd = torch.CudaTensor(numInstances)
    for i = 1, numInstances do
        dotProd[i] = torch.dot(roundedOutput[i], labels[i])
    end
    zero = torch.zeros(numInstances):cuda()
    numCorrect = dotProd:gt(zero):sum()
    accuracy = numCorrect * 100 / numInstances
    
    return accuracy
end

function calcClassAccuracy(classifier, data, labels)

    roundedOutput = classifier:forward(data):round()

    numInstances = data:size(1)
    dotProd = torch.CudaTensor(numInstances)
    for i = 1, numInstances do
        dotProd[i] = torch.dot(roundedOutput[i], labels[i])
    end
    zero = torch.zeros(numInstances):cuda()
    numCorrect = dotProd:gt(zero):sum()
    accuracy = numCorrect * 100 / numInstances
    
    return accuracy
end