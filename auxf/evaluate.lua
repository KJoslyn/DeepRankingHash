
function calcAndPrintHammingAccuracy(trainBatch, similarity_target, statsFile)

    imHash = getHashCodes(trainBatch.data[I])
    teHash = getHashCodes(trainBatch.data[X])

    s = similarity_target:size(1)

    for i = 4,8 do
        similarity = torch.eq(imHash, teHash):sum(2):ge(i)
        numCorrect = torch.eq(similarity, similarity_target):sum()
        statsPrint(string.format("Accuracy%d = %.2f", i, numCorrect*100 / s), statsFile)
    end
end

function getQueryAndDBCodes(fromModality, toModality, trainOrVal)

    -- local numQueries = 5000
    -- local numDatabase = 5000

    local endIdx = nil
    local imageIdxSet = nil
    local textIdxSet = nil

    -- trainImages, trainTexts, valImages, and valTexts come from pickSubset.lua. They are the indices of the images and texts that are
    -- used in training or validation respectively. They index trainset[modality]. Only 5000 images and 5000 texts are used for training,
    -- and 1000 images and 1000 texts are used for validation.
    if trainOrVal == 'train' then
        endIdx = 5000
        imageIdxSet = trainImages
        textIdxSet = trainTexts
    elseif trainOrVal == 'val' then
        endIdx = 1000
        imageIdxSet = valImages
        textIdxSet = valTexts
    else
        print("Error: input to getQueryAndDBCodes must be \'train\' or \'val\'")
    end

    images = imageIdxSet[ {{ 1, endIdx }} ]:long()
    texts = textIdxSet[ {{ 1, endIdx }} ]:long()

    if fromModality == I then
        queries = trainset[I]:index(1, images)
        queryLabels = train_labels_image:float():index(1, images)
    else
        queries = trainset[X]:index(1, texts)
        queryLabels = train_labels_text:float():index(1, texts)
    end

    if toModality == I then
        database = trainset[I]:index(1, images)
        databaseLabels = train_labels_image:float():index(1, images)
    else
        database = trainset[X]:index(1, texts)
        databaseLabels = train_labels_text:float():index(1, texts)
    end

    queryCodes = getHashCodes(queries)
    databaseCodes = getHashCodes(database)

    return queryCodes, databaseCodes, queryLabels, databaseLabels
end

function getDistanceAndSimilarityForMAP(queryCodes, databaseCodes, queryLabels, databaseLabels)

    queryLabels = queryLabels:cuda()
    databaseLabels = databaseLabels:cuda()

    numQueries = queryCodes:size(1)
    numDB = databaseCodes:size(1)

    D = torch.CudaByteTensor(numQueries, numDB)
    S = torch.CudaTensor(numQueries, numDB)
    sumAPs = 0
    for q = 1,numQueries do
        queryCodeRep = torch.expand(queryCodes[q]:reshape(L,1), L, numDB):transpose(1,2)
        D[q] = torch.ne(queryCodeRep, databaseCodes):sum(2)

        queryLabelRep = torch.expand(queryLabels[q]:reshape(24,1), 24, numDB):transpose(1,2)
        S[q] = torch.cmul(queryLabelRep, databaseLabels):max(2)
    end

    return D, S
end

function calcMAP(fromModality, toModality, trainOrVal)

    K = 50

    queryCodes, databaseCodes, queryLabels, databaseLabels = getQueryAndDBCodes(fromModality, toModality, trainOrVal)

    -- Q = 1
    Q = queryCodes:size(1)
    sumAPs = 0
    for q = 1,Q do
        -- databaseCodes = torch.reshape(databaseCodes, 4000, L)
        if fromModality == X then
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1, 1)
        else
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1)
        end

        ne = torch.ne(query, databaseCodes):sum(2)
        ne = torch.reshape(ne, ne:size(1))
        topkResults, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

        ind2 = randSort(topkResults)
        -- topkResults_sorted, ind2 = torch.sort(topkResults)
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

function getHashCodes(data)

    return computeInBatches(calcHashCodes, torch.CudaLongTensor(data:size(1), L), data, nil)
end

function calcRoundedOutput(data) 
    -- This function requires a global variable called "imageClassifier" or "textClassifier"

    if data:size(2) == 3 then -- Image modality
        return imageClassifier:cuda():forward(data:cuda()):round()
    else -- Text modality
        return textClassifier:cuda():forward(data:cuda()):round()
    end
end

function calcHashCodes(data)

    if data:size(2) == 3 then -- Image modality
        pred = imageHasher:forward(data:cuda())
    else -- Text modality
        pred = textHasher:forward(data:cuda())
    end

    reshapedInput = nn.Reshape(L,k):cuda():forward(pred)
    maxs, indices = torch.max(reshapedInput, 3)
    return indices
end

function computeInBatches(computeFunction, output, data) 

    if data:size(2) == 1075 then -- Text modality
        return computeFunction(data)
    elseif data:size(2) == 3 then -- Image modality
        N = data:size(1)
        local batchSize = 128
        local numBatches = torch.ceil(N / batchSize)
        for b = 0, numBatches - 1 do
            startIndex = b * batchSize + 1
            endIndex = math.min((b + 1) * batchSize, N)
            batch = data[{{ startIndex, endIndex }}]
            output[{{ startIndex, endIndex}}] = computeFunction(batch)
        end
        return output    
    else
        print("Error: unrecognized modality")
    end
end

function calcClassAccuracy(data, labels)

    roundedOutput = computeInBatches(calcRoundedOutput, torch.CudaTensor(data:size(1), 24), data)

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

function calcClassAccuracyForModality(modality)
    if modality == I then
        data = trainset[I]:index(1, trainImages)
        labels = train_labels_image:float():index(1, trainImages):cuda() -- TODO: is float() necessary?
    else
        data = trainset[X]:index(1, trainTexts)
        labels = train_labels_text:float():index(1, trainTexts):cuda()
    end
    return calcClassAccuracy(data, labels)
end

function statsPrint(line, statsFile1, statsFile2)
    if statsFile1 then
        statsFile1:write(line .. "\n")
    end
    if statsFile2 then
        statsFile2:write(line .. "\n")
    end
    print(line)
end

function randSort(vals)

    local edges = {}
    edges[1] = 1
    local i = 2
    for i = 2,vals:size(1) do
        if vals[i] > vals[i-1] then
            edges[#edges+1] = i
        end
    end
    edges[#edges+1] = vals:size(1) + 1

    local ind = torch.Tensor()
    for e = 2,#edges do
        local perm = torch.randperm(edges[e] - edges[e-1])
        local perm = perm:add(edges[e-1] - 1)
        ind = torch.cat(ind, perm)
    end

    return ind:long()
end

function getHashCodeBitCounts()

    local th = getHashCodes(trainset[X])
    local ih = getHashCodes(trainset[I])
    local ibc = torch.LongTensor(L,k)
    local tbc = torch.LongTensor(L,k)
    for i=1,L do
        for j=1,k do
            ibc[i][j] = torch.eq(ih:select(2,i),j):sum()
            tbc[i][j] = torch.eq(th:select(2,i),j):sum()
        end
    end

    hbc = {}
    hbc[I] = ibc
    hbc[X] = tbc
    return hbc
end