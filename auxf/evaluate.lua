
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
        data = d.trainset[I].data
        labels = d.trainset[I].label:float()
    else
        data = d.trainset[X].data
        labels = d.trainset[X].label:float()
    end
    return getClassAccuracy(data, labels:cuda())
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

    if data:size(2) == 3 then -- Image modality
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
    else -- Text modality
        return computeFunction(data)
    end
end

-- //////////////////////////////
-- mAP Functions
-- /////////////////////////////

local function getClassesAndQuantity(classArg)

    local classes
    if type(classArg) == 'string' then
        classes = {classArg}
    elseif type(classArg) == 'table' then
        classes = classArg
    end

    local quantity = 0
    local includePretraining = false
    for c = 1, #classes do
        local class = classes[c]
        if class == 'training' then
            quantity = quantity + d.dataset:sizeTrain()
        elseif class == 'pretraining' then
            quantity = quantity + d.dataset:sizePretraining()
            includePretraining = true
        elseif class == 'val' then
            quantity = quantity + d.dataset:sizeVal()
        elseif class == 'query' then
            quantity = quantity + d.datatset:sizeTest()
        end
    end

    return classes, quantity, includePretraining
end

function getImageCodesAndLabels(classArg)

    local classes, N, includePretraining = getClassesAndQuantity(classArg)

    local codes = torch.CudaLongTensor(N, p.L)
    local labels = torch.FloatTensor(N, p.numClasses)

    local allData = torch.FloatTensor() -- TODO: Correct?
    local allLabels = torch.FloatTensor()
    if not includePretraining then
        for c = 1, #classes do
            local catData, catLabels
            local class = classes[c]
            if class == 'training' then
                catData = d.trainset[I].data
                catLabels = d.trainset[I].label
            elseif class == 'val' then
                catData = d.valset[I].data
                catLabels = d.valset[I].label
            elseif class == 'query' then
                catData = d.testset[I].data
                catLabels = d.testset[I].label
            end
            allData = torch.cat(allData, catData, 1)
            allLabels = torch.cat(allLabels, catLabels, 1)
        end
    end

    local batchSize = 128
    local numBatches = math.ceil(N / batchSize)

    for batchNum = 0, numBatches - 1 do

        local startIndex = batchNum * batchSize + 1
        local endIndex = math.min((batchNum + 1) * batchSize, N)

        local batchData, batchLabels
        if includePretraining then
            batchData, batchLabels = d.dataset:getBySplit(classes, 'I', startIndex, endIndex)
        else
            batchData = allData[{ {startIndex, endIndex} }]
            batchLabels = allLabels[{ {startIndex, endIndex} }]
        end

        codes[{ {startIndex, endIndex} }] = calcHashCodes(batchData):reshape(endIndex - startIndex + 1, p.L)
        labels[{ {startIndex, endIndex} }] = batchLabels
    end

    return codes, labels
end

function getTextCodesAndLabels(classArg)

    local classes, quantity, includePretraining = getClassesAndQuantity(classArg)

    local tags, labels = d.dataset:getBySplit(classes, 'X', 1, quantity)
    local codes = calcHashCodes(tags):reshape(quantity, p.L)

    return codes, labels
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

function calcMAP(modalityFrom, modalityTo, classesFrom, classesTo)

    m.fullModel:evaluate()

    local queryCodes, queryLabels, dbCodes, dbLabels
    if modalityFrom == I then
        queryCodes, queryLabels = getImageCodesAndLabels(classesFrom)
        dbCodes, dbLabels = getTextCodesAndLabels(classesTo)
    else
        queryCodes, queryLabels = getTextCodesAndLabels(classesFrom)
        dbCodes, dbLabels = getImageCodesAndLabels(classesTo)
    end

    if saveToMatFile then
        local D, S = getDistanceAndSimilarityForMAP(queryCodes, dbCodes, queryLabels, dbLabels)
        saveDistAndSimToMatFile(D,S)
    end

    return calcMAP_old(queryCodes, dbCodes, queryLabels, dbLabels)
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

function getHashCodeBitCounts(trainset)

    local th = getHashCodes(trainset[X].data)
    local ih = getHashCodes(trainset[I].data)
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

    local pred = getRawPredictions(d.trainset[modality].data)
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

function plotClassAccAndLoss(epoch, plotLoss)
    local elapsedEpochs = epoch - (g.plotStartEpoch - 1) -- startEpoch should be one greater than a multiple of g.plotNumEpochs
    if elapsedEpochs / g.plotNumEpochs > 1 then
        local x = torch.linspace(g.plotStartEpoch + g.plotNumEpochs - 1, epoch, elapsedEpochs / g.plotNumEpochs)

        local y = g.avgDataAcc
        local yh = g.maxDataAcc
        local yl = g.minDataAcc
        local yy = torch.cat(x,yh,2)
        local yy = torch.cat(yy,yl,2)

        local y2 = g.avgDataLoss
        local yh2 = g.maxDataLoss
        local yl2 = g.minDataLoss
        local yy2 = torch.cat(x, yh2, 2)
        local yy2 = torch.cat(yy2, yl2, 2)

        if not g.plotFilename then
            local date = os.date("*t", os.time())
            local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
            g.plotFilename = g.snapshotDir .. '/' .. dateStr .. '_' .. 'plot.pdf' 
        end
        gnuplot.pdffigure(g.plotFilename)
        if plotLoss then
            gnuplot.plot({yy2,'with filledcurves fill transparent solid 0.5'},{x,yl2,'with lines ls 1'},{x,yh2,'with lines ls 1'},{x,y2,'with lines ls 1'},
                         {yy,'with filledcurves fill transparent solid 0.5'},{x,yl,'with lines ls 1'},{x,yh,'with lines ls 1'},{x,y,'with lines ls 1'})
        else
            gnuplot.plot({yy,'with filledcurves fill transparent solid 0.5'},{x,yl,'with lines ls 1'},{x,yh,'with lines ls 1'},{x,y,'with lines ls 1'})
        end
        gnuplot.plotflush()
    end
end