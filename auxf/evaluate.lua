
-- //////////////////////////////
-- These functions use computeInBatches with a "calc" function
-- /////////////////////////////

function getRawPredictions(data)

    return computeInBatches(calcRawPredictions, torch.CudaTensor(data:size(1), p.L*p.k), data, nil)
end

function getHashCodes(data)

    return computeInBatches(calcHashCodes, torch.CudaLongTensor(data:size(1), p.L), data, nil)
end

function getClassAccuracy(data, labels)

    roundedOutput = computeInBatches(calcRoundedClassifierOutput, torch.CudaTensor(data:size(1), p.numClasses), data)

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

function getClassAccuracyForModality(modality)
    if modality == I then
        data = d.trainset[I]:index(1, d.trainImages)
        labels = d.train_labels_image:float():index(1, d.trainImages):cuda() -- TODO: is float() necessary?
    else
        data = d.trainset[X]:index(1, d.trainTexts)
        labels = d.train_labels_text:float():index(1, d.trainTexts):cuda()
    end
    return getClassAccuracy(data, labels)
end

-- //////////////////////////////
-- These functions are to be used as input functions to computeInBatches
-- /////////////////////////////

function calcRoundedClassifierOutput(data) 

    if data:size(2) == 3 then -- Image modality
        return m.imageClassifier:cuda():forward(data:cuda()):round()
    else -- Text modality
        return m.textClassifier:cuda():forward(data:cuda()):round()
    end
end

function calcRawPredictions(data)

    if data:size(2) == 3 then -- Image modality
        return m.imageHasher:forward(data:cuda())
    else -- Text modality
        return m.textHasher:forward(data:cuda())
    end
end

function calcHashCodes(data)

    pred = calcRawPredictions(data)

    reshapedInput = nn.Reshape(p.L,p.k):cuda():forward(pred)
    maxs, indices = torch.max(reshapedInput, 3)
    return indices
end

function computeInBatches(computeFunction, output, data) 

    if data:size(2) == 1075 then -- Text modality
        return computeFunction(data)
    elseif data:size(2) == 3 then -- Image modality
        local N = data:size(1)
        local batchSize = 128
        local numBatches = torch.ceil(N / batchSize)
        for b = 0, numBatches - 1 do
            local startIndex = b * batchSize + 1
            local endIndex = math.min((b + 1) * batchSize, N)
            batch = data[{{ startIndex, endIndex }}]
            output[{{ startIndex, endIndex}}] = computeFunction(batch)
        end
        return output    
    else
        print("Error: unrecognized modality")
    end
end

-- //////////////////////////////
-- mAP Functions
-- /////////////////////////////

function getQueryAndDBCodes(fromModality, toModality, trainOrVal)

    local endIdx = nil
    local imageIdxSet = nil
    local textIdxSet = nil

    -- d.trainImages, d.trainTexts, d.valImages, and d.valTexts come from pickSubset.lua. They are the indices of the images and texts that are
    -- used in training or validation respectively. They index d.trainset[modality]. Only 5000 images and 5000 texts are used for training,
    -- and 1000 images and 1000 texts are used for validation.
    if trainOrVal == 'train' then
        endIdx = d.trainImages:size(1)
        imageIdxSet = d.trainImages
        textIdxSet = d.trainTexts
    elseif trainOrVal == 'val' then
        endIdx = d.valImages:size(1)
        imageIdxSet = d.valImages
        textIdxSet = d.valTexts
    else
        print("Error: input to getQueryAndDBCodes must be \'train\' or \'val\'")
    end

    images = imageIdxSet[ {{ 1, endIdx }} ]:long()
    texts = textIdxSet[ {{ 1, endIdx }} ]:long()

    if fromModality == I then
        queries = d.trainset[I]:index(1, images)
        queryLabels = d.train_labels_image:float():index(1, images)
    else
        queries = d.trainset[X]:index(1, texts)
        queryLabels = d.train_labels_text:float():index(1, texts)
    end

    if toModality == I then
        database = d.trainset[I]:index(1, images)
        databaseLabels = d.train_labels_image:float():index(1, images)
    else
        database = d.trainset[X]:index(1, texts)
        databaseLabels = d.train_labels_text:float():index(1, texts)
    end

    queryCodes = getHashCodes(queries)
    databaseCodes = getHashCodes(database)

    return queryCodes, databaseCodes, queryLabels, databaseLabels
end

function getQueryAndDBCodesTest(fromModality, toModality)

    if fromModality == I then
        queries = d.testset[I]
        queryLabels = d.test_labels_image:float()
    else
        queries = d.testset[X]
        queryLabels = d.test_labels_text:float()
    end

    if toModality == I then
        database = d.trainset[I]
        databaseLabels = d.train_labels_image:float()
    else
        database = d.trainset[X]
        databaseLabels = d.train_labels_text:float()
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
        queryCodeRep = torch.expand(queryCodes[q]:reshape(p.L,1), p.L, numDB):transpose(1,2)
        D[q] = torch.ne(queryCodeRep, databaseCodes):sum(2)

        queryLabelRep = torch.expand(queryLabels[q]:reshape(p.numClasses,1), p.numClasses, numDB):transpose(1,2)
        S[q] = torch.cmul(queryLabelRep, databaseLabels):max(2)
    end

    return D, S
end

function saveDistAndSimToMatFile(D,S)

    local D_new = torch.LongTensor(D:size(1),D:size(2)):copy(D)
    local S_new = torch.LongTensor(S:size(1),S:size(2)):copy(S)
    if not matio then
        matio = require 'matio'
    end
    local date = os.date("*t", os.time())
    local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
    matio.save(g.snapshotDir .. '/DS_data_' .. dateStr .. '.mat', {D=D_new,S=S_new})
end

function calcMAP(fromModality, toModality, trainValOrTest, saveToMatFile)

    m.fullModel:evaluate()

    if trainValOrTest == 'test' then
        queryCodes, databaseCodes, queryLabels, databaseLabels = getQueryAndDBCodesTest(fromModality, toModality, false)
    else
        queryCodes, databaseCodes, queryLabels, databaseLabels = getQueryAndDBCodes(fromModality, toModality, trainValOrTest)
    end

    databaseCodes = databaseCodes:reshape(databaseCodes:size(1), databaseCodes:size(2))
    queryCodes = queryCodes:reshape(queryCodes:size(1), queryCodes:size(2))

    if saveToMatFile then
        local D, S = getDistanceAndSimilarityForMAP(queryCodes, databaseCodes, queryLabels, databaseLabels)
        saveDistAndSimToMatFile(D,S)
    end

    -- return compute_map(queryCodes, databaseCodes, S)
    return calcMAP_old(queryCodes, databaseCodes, queryLabels, databaseLabels)
end

function compute_map(test_data, train_data, similarities)

  local test_codes = test_data:cuda()
  local train_codes = train_data:cuda()
  local smat = similarities:cuda()

  local ntest = test_codes:size()[1]
  local ntrain = train_codes:size()[1]

  local pdist = -2*torch.mm(test_codes, train_codes:t())

  pdist:add(train_codes:norm(2, 2):pow(2):t():expandAs(pdist))

  local cind, rind, lind 
  _, cind = torch.sort(pdist, 2)

  rind = torch.linspace(1, ntest, ntest):repeatTensor(ntrain, 1):t():csub(1):long()

  lind = torch.add(rind*ntrain, cind:long()):view(ntest*ntrain)
  
  local resmat, resmat_cumsum, prec_mat

  resmat = smat:view(ntrain*ntest):index(1, lind):view(ntest, ntrain)

  resmat_cumsum = resmat:cumsum(2)

  prec_mat = torch.cdiv(resmat_cumsum, 
    torch.linspace(1, ntrain, ntrain):view(1, ntrain):expandAs(resmat):cuda())

  local mAP = torch.cmul(prec_mat, resmat):sum(2):cdiv(resmat:sum(2)):mean()

  return mAP

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

-- //////////////////////////////
-- Regularizer evaluation functions
-- /////////////////////////////

function getHashCodeBitCounts(data)

    -- data should be d.trainset or trainBatch.data

    local th = getHashCodes(data[X])
    local ih = getHashCodes(data[I])
    local ibc = torch.LongTensor(p.L,p.k)
    local tbc = torch.LongTensor(p.L,p.k)
    for i=1,p.L do
        for j=1,p.k do
            ibc[i][j] = torch.eq(ih:select(2,i),j):sum()
            tbc[i][j] = torch.eq(th:select(2,i),j):sum()
        end
    end

    hbc = {}
    hbc[I] = ibc
    hbc[X] = tbc
    stdev_image = ibc:double():std()
    stdev_text = tbc:double():std()
    return hbc, stdev_image, stdev_text
end

function getSoftMaxAvgDistFromOneHalf(modality)

    local pred = getRawPredictions(d.trainset[modality])
    pred = pred:view(-1)
    local avgDist = torch.abs(pred - 0.5):mean()

    return avgDist
end

-- //////////////////////////////
-- Miscellaneous function 
-- /////////////////////////////

function statsPrint(line, statsFile1, statsFile2)
    if statsFile1 then
        statsFile1:write(line .. "\n")
    end
    if statsFile2 then
        statsFile2:write(line .. "\n")
    end
    print(line)
end

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
