
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

    local roundedOutput = computeInBatches(calcRoundedClassifierOutput, torch.CudaTensor(data:size(1), p.numClasses), data)

    local numInstances = data:size(1)
    local dotProd = torch.CudaTensor(numInstances)
    for i = 1, numInstances do
        dotProd[i] = torch.dot(roundedOutput[i], labels[i])
    end
    local zero = torch.zeros(numInstances):cuda()
    local numCorrect = dotProd:gt(zero):sum()
    local accuracy = numCorrect * 100 / numInstances
    
    return accuracy
end

function getClassAccuracyForModality(modality)
    local data = d.trainset[modality].data
    local labels = d.trainset[modality].label:float()
    return getClassAccuracy(data, labels:cuda())
end

-- //////////////////////////////
-- These functions are to be used as input functions to computeInBatches
-- /////////////////////////////

function calcRoundedClassifierOutput(data) 

    local cData = torch.CudaTensor(data:size()):copy(data)
    -- local z
    if data:size(2) == 3 then -- Image modality
        -- return m.imageClassifier:cuda():forward(data:cuda()):round()
        return m.imageClassifier:forward(cData):round()
    else -- Text modality
        return m.textClassifier:forward(cData):round()
    end
    -- cData = nil
    -- collectgarbage()
    -- return z
end

function calcRawPredictions(data)

    if data:size(2) == 3 or data:size(2) == 4096 then -- Image modality. First is raw pixels, second is alexNet features
        return m.imageHasher:forward(data:cuda())
    else -- Text modality
        return m.textHasher:forward(data:cuda())
    end
end

function calcHashCodes(data)

    local pred = calcRawPredictions(data)

    local reshapedInput = nn.Reshape(p.L,p.k):cuda():forward(pred)
    local maxs, indices = torch.max(reshapedInput, 3)
    -- indices = indices:reshape(indices:size(1), indices:size(2))
    -- indices = torch.LongTensor(indices:size(1), indices:size(2)):copy(indices)
    return indices
end

function computeInBatches(computeFunction, output, data) 

    -- local timer = torch.Timer()
    -- local tt = torch.Timer()
    -- tt:stop()
    local N = data:size(1)

    -- Don't make the batch size too big. Forward propagation of a large batch eats up a lot of GPU space that we can't reclaim.
    -- It might be related to the spatial convolution layers https://groups.google.com/forum/#!topic/torch7/krvJP5fqTgQ
    -- Also related- env variable THC_CACHING_ALLOCATOR=0 allows collectgarbage() to work as expected
    -- https://github.com/torch/torch7/issues/229
    local batchSize = 128 
    local numBatches = torch.ceil(N / batchSize)
    for b = 0, numBatches - 1 do
        local startIndex = b * batchSize + 1
        local endIndex = math.min((b + 1) * batchSize, N)
        local batch = data[{{ startIndex, endIndex }}]
        -- output[{{ startIndex, endIndex}}] = computeFunction(batch)
        output[{{ startIndex, endIndex}}]:copy(computeFunction(batch))
        if b % 10 == 0 then
            -- tt:resume()
            collectgarbage()
            -- tt:stop()
        end
    end
    -- timer:stop()
    -- print(string.format("Time in compute in batches: %.2f", timer:time().real))
    -- print(string.format("Time collecting garbage: %.2f", tt:time().real))
    return output    
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
        if batchNum % 10 == 0 then
            collectgarbage()
        end
    end

    return codes, labels
end

function getCodesAndLabelsForModalityAndClass(modality, class, usePrecomputedCodes)

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
    end

    if class == 'pretraining' and p.datasetType == 'nus' and not p.usePretrainedImageFeatures then
        split.codes, split.label = calcPretrainsetHashCodesForNuswide(modalityChar)
        return split.codes, split.label
    end

    if class == 'pretraining' and not d.pretrainset[modality].data then -- assumes p.datasetType == 'mir'
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
    -- local allCodes = torch.LongTensor()
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

    print("Calculating D and S matrices for matlab MAP")

    local D_timer = torch.Timer()
    D_timer:stop()
    local S_timer = torch.Timer()
    S_timer:stop()

    local queryLabels = queryLabels:cuda()
    local databaseLabels = databaseLabels:cuda()

    local numQueries = queryCodes:size(1)
    local numDB = databaseCodes:size(1)

    local D = torch.CudaByteTensor(numQueries, numDB)
    local S = torch.CudaTensor(numQueries, numDB)
    local sumAPs = 0
    for q = 1,numQueries do
        D_timer:resume()
        local queryCodeRep = torch.expand(queryCodes[q]:reshape(p.L,1), p.L, numDB):transpose(1,2)
        D[q] = torch.ne(queryCodeRep, databaseCodes):sum(2)
        D_timer:stop()

        S_timer:resume()
        local queryLabelRep = torch.expand(queryLabels[q]:reshape(p.numClasses,1), p.numClasses, numDB):transpose(1,2)
        S[q] = torch.cmul(queryLabelRep, databaseLabels):max(2)
        S_timer:stop()
    end
    print(string.format("D calc time: %.2f", D_timer:time().real))
    print(string.format("S calc time: %.2f", S_timer:time().real))

    return D, S
end

function saveDistAndSimToMatFile(D,S,DS_arg)

    local D_new = torch.LongTensor(D:size(1),D:size(2)):copy(D)
    local S_new = torch.LongTensor(S:size(1),S:size(2)):copy(S)
    if not matio then
        matio = require 'matio'
    end
    local filename
    if type(DS_arg) == 'boolean' then
        local date = os.date("*t", os.time())
        local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
        filename = g.snapshotDir .. '/DS_data_' .. dateStr .. '.mat'
    elseif type(DS_arg) == 'string' then
        filename = DS_arg
    end
    matio.save(filename, {D=D_new,S=S_new})
end

function calcMAP(modalityFrom, modalityTo, classesFrom, classesTo, usePrecomputedCodes, DS_arg)

    local timer = torch.Timer()
    timer:resume()

    m.fullModel:evaluate()

    local queryCodes, queryLabels = getCodesAndLabels(modalityFrom, classesFrom, usePrecomputedCodes)
    local dbCodes, dbLabels = getCodesAndLabels(modalityTo, classesTo, usePrecomputedCodes)

    if DS_arg then
        local D, S = getDistanceAndSimilarityForMAP(queryCodes, dbCodes, queryLabels, dbLabels)
        saveDistAndSimToMatFile(D,S,DS_arg)
    end

    local map = calcMAP_old(queryCodes, dbCodes, queryLabels, dbLabels)
    -- D, S = getDistanceAndSimilarityForMAP(queryCodes, dbCodes, queryLabels, dbLabels)
    -- local map = compute_map(dbCodes, queryCodes, S)

    timer:stop()
    local compute_time = timer:time().real
    return map, compute_time
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

    hbc = {} -- TODO: Make local? Nice to have globally to look at though
    hbc[I] = ibc
    hbc[X] = tbc
    local stdev_image = ibc:double():std()
    local stdev_text = tbc:double():std()
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

    local imHash = getHashCodes(trainBatch.data[I])
    local teHash = getHashCodes(trainBatch.data[X])

    local s = similarity_target:size(1)

    for i = 4,8 do
        local similarity = torch.eq(imHash, teHash):sum(2):ge(i)
        local numCorrect = torch.eq(similarity, similarity_target):sum()
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