
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

function calcPretrainsetHashCodesForNuswide(modalityChar)

    local N = d.dataset:sizePretraining()

    local codes = torch.CudaLongTensor(N, p.L)
    local labels = torch.FloatTensor(N, p.numClasses)

    local batchSize = 128
    local numBatches = math.ceil(N / batchSize)

    for batchNum = 0, numBatches - 1 do

        local startIndex = batchNum * batchSize + 1
        local endIndex = math.min((batchNum + 1) * batchSize, N)

        local batchData, batchLabels = d.dataset:getBySplit('pretraining', modalityChar, startIndex, endIndex)
        codes[{ {startIndex, endIndex} }] = calcHashCodes(batchData):reshape(endIndex - startIndex + 1, p.L)
        labels[{ {startIndex, endIndex} }] = batchLabels
    end

    return codes, labels
end

function getCodesAndLabelsForModalityAndClass(modality, class, usePrecomputedCodes)

    if class == 'pretraining' and p.datasetType == 'nus' then
        return calcPretrainsetHashCodesForNuswide(modalityChar)
    end

    local split, modalityChar
    if class == 'pretraining' then
        split = d.pretrainset
        if modality == I then
            modalityChar = 'I'
        else
            modalityChar = 'X'
        end
    elseif class == 'training' then
        split = d.trainset
    elseif class == 'val' then
        split = d.valset
    elseif class == 'query' then
        split = d.testset
    end
    split = split[modality]

    if usePrecomputedCodes and split.codes then
        return split.codes, split.label
    elseif class == 'pretraining' and not d.pretrainset[modality].data then -- assumes p.datasetType == 'mir'
        d.pretrainset[modality].data, d.pretrainset[modality].label = d.dataset:getBySplit('pretraining', modalityChar, 1, d.dataset:sizePretraining())
    end
    local data = split.data

    -- Compute and save the computed hash codes
    split.codes = getHashCodes(data):reshape(data:size(1), p.L) -- TODO: Check if this works

    return split.codes, split.label
end

function getCodesAndLabels(modality, classArg, usePrecomputedCodes)

    local classes
    if type(classArg) == 'string' then
        classes = {classArg}
    elseif type(classArg) == 'table' then
        classes = classArg
    end
    -- local codes = torch.CudaLongTensor(N, p.L)
    -- local labels = torch.FloatTensor(N, p.numClasses)

    -- local allData = torch.FloatTensor() -- TODO: Correct?
    local allCodes = torch.CudaLongTensor()
    local allLabels = torch.FloatTensor()
    for c = 1, #classes do
        local class = classes[c]
        local catCodes, catLabels = getCodesAndLabelsForModalityAndClass(modality, class, usePrecomputedCodes)
        allCodes = torch.cat(allCodes, catCodes, 1)
        allLabels = torch.cat(allLabels, catLabels, 1)
    end

    return allCodes, allLabels
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

function calcMAP(modalityFrom, modalityTo, classesFrom, classesTo, usePrecomputedCodes)

    m.fullModel:evaluate()

    local queryCodes, queryLabels = getCodesAndLabels(modalityFrom, classesFrom, usePrecomputedCodes)
    local dbCodes, dbLabels = getCodesAndLabels(modalityTo, classesTo, usePrecomputedCodes)

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

local function normalizePlotLine(line)

    local norm = line / line:max()
    return norm
end

local function getPlotLines(epoch)

    if not g.plotStartEpoch then
        g.plotStartEpoch = epoch
    end

    local elapsedEpochs = epoch - (g.plotStartEpoch - 1) -- startEpoch should be one greater than a multiple of g.plotNumEpochs

    local x = torch.linspace(g.plotStartEpoch, epoch, elapsedEpochs)
    local y_loss = normalizePlotLine(s.y_loss)
    local y_cross = normalizePlotLine(s.y_cross)
    local y_b1 = normalizePlotLine(s.y_b1)
    local y_b2 = normalizePlotLine(s.y_b2)
    local y_q1 = normalizePlotLine(s.y_q1)
    local y_q2 = normalizePlotLine(s.y_q2)
    local y_ixt = s.y_ixt
    local y_xit = s.y_xit
    local y_ixv = s.y_ixv
    local y_xiv = s.y_xiv

    local lines = {}
    local bw = m.criterion.weights[2]
    local qw = m.criterion.weights[4]
    
    if bw > 0 or qw > 0 then
        lines[#lines + 1] = {x,y_loss, 'with lines ls 1 lc rgb \'black\''}
    end
    lines[#lines + 1] = {x,y_cross, 'with lines ls 1 lc rgb \'blue\''}
    if bw > 0 then
        local y_bavg = (y_b1 + y_b2) / 2
        -- lines[#lines + 1] = {x,y_b1, 'with lines ls 2 lc rgb \'green\''}
        -- lines[#lines + 1] = {x,y_b2, 'with lines ls 2 lc rgb \'purple\''}
        lines[#lines + 1] = {x,y_bavg, 'with lines ls 2 lc rgb \'yellow\''}
    end
    if qw > 0 then
        lines[#lines + 1] = {x,y_q1, 'with lines ls 1 lc rgb \'red\''}
        lines[#lines + 1] = {x,y_q2, 'with lines ls 1 lc rgb \'orange\''}
    end
    if y_ixt:dim() > 0 then
        lines[#lines + 1] = {x,y_ixt, 'with lines ls 2 lc rgb \'blue\''}
        lines[#lines + 1] = {x,y_xit, 'with lines ls 2 lc rgb \'green\''}
        lines[#lines + 1] = {x,y_ixv, 'with lines ls 2 lc rgb \'purple\''}
        lines[#lines + 1] = {x,y_xiv, 'with lines ls 2 lc rgb \'red\''}
    end

    return lines
end

function plotCrossModalLoss(epoch)

    local lines = getPlotLines(epoch)

    if not g.plotFilename then
        local date = os.date("*t", os.time())
        local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
        g.plotFilename = g.snapshotDir .. '/' .. dateStr .. '_' .. 'CMplot.pdf' 
    end

    gnuplot.pdffigure(g.plotFilename)

    gnuplot.plot(unpack(lines))

    gnuplot.raw('set style line 2 dashtype 2')

    gnuplot.plotflush()
end

function plotCrossModalLoss_old(epoch)

    if not g.plotStartEpoch then
        g.plotStartEpoch = epoch
    end

    local elapsedEpochs = epoch - (g.plotStartEpoch - 1) -- startEpoch should be one greater than a multiple of g.plotNumEpochs

    local x = torch.linspace(g.plotStartEpoch, epoch, elapsedEpochs)
    local y_loss = s.y_loss
    local y_ixt = s.y_ixt
    local y_xit = s.y_xit
    local y_ixv = s.y_ixv
    local y_xiv = s.y_xiv

    if not g.plotFilename then
        local date = os.date("*t", os.time())
        local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
        g.plotFilename = g.snapshotDir .. '/' .. dateStr .. '_' .. 'CMplot.pdf' 
    end

    gnuplot.pdffigure(g.plotFilename)

    if y_ixt:dim() == 0 then
        gnuplot.plot({x,y_loss, 'with lines ls 1 lc rgb \'black\''})
    else
        gnuplot.plot({x,y_loss, 'with lines ls 1 lc rgb \'black\''},
                    {x,y_cross, 'with lines ls 1 lc rgb \'blue\''},
                    {x,y_b1, 'with lines ls 1 lc rgb \'green\''},
                    {x,y_b2, 'with lines ls 1 lc rgb \'purple\''},
                    {x,y_q1, 'with lines ls 1 lc rgb \'red\''},
                    {x,y_q2, 'with lines ls 1 lc rgb \'orange\''},
                    {x,y_ixt, 'with lines ls 1 lc rgb \'blue\''},
                    {x,y_xit, 'with lines ls 1 lc rgb \'green\''},
                    {x,y_ixv, 'with lines ls 1 lc rgb \'purple\''},
                    {x,y_xiv, 'with lines ls 1 lc rgb \'red\''})
    end

    gnuplot.plotflush()
end

function plotClassAccAndLoss(epoch, plotLoss)
    local elapsedEpochs = epoch - (g.plotStartEpoch - 1) -- startEpoch should be one greater than a multiple of g.plotNumEpochs
    if elapsedEpochs / g.plotNumEpochs > 1 then
        local x = torch.linspace(g.plotStartEpoch + g.plotNumEpochs - 1, epoch, elapsedEpochs / g.plotNumEpochs)

        local y = s.avgDataAcc
        local yh = s.maxDataAcc
        local yl = s.minDataAcc
        local yy = torch.cat(x,yh,2)
        local yy = torch.cat(yy,yl,2)

        local y2 = s.avgDataLoss
        local yh2 = s.maxDataLoss
        local yl2 = s.minDataLoss
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