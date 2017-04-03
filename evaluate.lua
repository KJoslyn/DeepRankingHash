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

function calcMAP(fromModality, toModality, trainBatch, batchIdx) -- TODO: Remove 3rd and 4th parameters

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
        topkResults, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

        topkResults_sorted, ind2 = torch.sort(topkResults)
        topkIndices = ind:index(1,ind2)

        qLabel = queryLabels[q]

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
        if correct > 0 then -- Correct should only be 0 if there are a small # of database objects and/or poorly trained
            AP = AP / correct -- Recall component (divide by number of ground truth positives in top k)
        end
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