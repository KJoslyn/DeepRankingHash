require 'fullNet'

function runAllParamsets(datasetType, paramFactorialSet, numEpochs, evalInterval, iterationsPerEpoch, usePretrainedImageFeatures, consecutiveStop)

    -- This is the main function to call

    -- TODO: This is set to a constant
    -- local iterationsPerEpoch = 25

    loadParamsAndPackages(datasetType, iterationsPerEpoch, usePretrainedImageFeatures)

    local autoStatsDir
    if datasetType == 'mir' then
        autoStatsDir = 'mirflickr'
    elseif datasetType == 'nus' then
        autoStatsDir = 'nuswide'
    end
    autoStatsDir = autoStatsDir .. '/CM'

    g.statsDir = '/home/kjoslyn/kevin/Project/autoStats/' .. autoStatsDir
    g.meta = io.open(g.statsDir .. "/metaStats.txt", 'a')
    g.startStatsId = nil

    p.numEpochs = numEpochs
    p.evalInterval = evalInterval
    if consecutiveStop then
        p.consecutiveStop = consecutiveStop
    else
        p.consecutiveStop = numEpochs + 1
    end

    local numParamCombs = getNumParamCombs(paramFactorialSet)
    p.numKFoldSplits = getNumKFoldSplits(paramFactorialSet)

    g.paramSettingsLegend = {}
    -- TODO: Convert to concatenating tensors (due to validating params)
    g.trainResultsMatrix = torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
    g.valResultsMatrix = torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
    g.resultsParamIdx = 0

    recursiveRunAllParamsets(paramFactorialSet, paramFactorialSet, 0, #paramFactorialSet)

    io.close(g.meta)
end

function recursiveRunAllParamsets(pfs_part, pfs_full, paramCount, numParams)

    if paramCount == numParams then
        -- printParams(pfs_full)
        if validateParams() then
            -- If this starts a new parameter combination, increment the paramIdx (initialized to 0)
            if p.numKFoldSplits == 1 or p.kFoldNum == 1 then
                g.resultsParamIdx = g.resultsParamIdx + 1
            end
            runWithParams(pfs_full)
        end
    else
        local thisParamset = pfs_part[1]
        local paramName = thisParamset[1]
        local valueSet = thisParamset[2]
        for _, value in pairs(valueSet) do

            local oldValue = setParamValue(paramName, value)
            recursiveRunAllParamsets( { unpack(pfs_part, 2, #pfs_part) }, pfs_full, paramCount + 1, numParams)
            setParamValue(paramName, oldValue)
        end
    end
end

function validateParams()

    if p.quantRegWeight > 0 and p.balanceRegWeight == 0 then
        return false
    -- elseif p.L * math.log(p.k) / math.log(2) ~= 60 then
    --     return false
    else
        return true
    end
end

function getNumParamCombs(pfs)

    -- Gets the number of the unique parameter combinations in the paramFactorialSet.
    -- Ignores the number of k-fold splits

    local count = 1
    for i = 1,#pfs do
        if pfs[i][1] ~= 'kfn' then
            count = count * #pfs[i][2]
        end
    end
    return count
end

function getNumKFoldSplits(pfs)

    for i = 1,#pfs do
        if pfs[i][1] == 'kfn' then
            return #pfs[i][2]
        end
    end
    return 1
end

function setParamValue(paramName, value)

    -- lr = lrMultForHashLayer
    -- bw = bit balance regularizer weight
    -- qw = quantizer weight

    local longParamName = getLongParamName(paramName)
    local oldValue = p[longParamName]
    p[longParamName] = value
    return oldValue
end

function getLongParamName(short)

    if short == 'lr' then
        return 'lrMultForHashLayer'
    elseif short == 'mt' then
        return 'modelType'
    elseif short == 'bw' then
        return 'balanceRegWeight'
    elseif short == 'qw' then
        return 'quantRegWeight'
    elseif short == 'lrd' then
        return 'baseLearningRateDecay'
    elseif short == 'wd' then
        return 'baseWeightDecay'
    elseif short == 'mom' then
        return 'baseMomentum'
    elseif short == 'L' then
        return 'L'
    elseif short == 'k' then
        return 'k'
    -- TODO, or not
    -- elseif short == 'IClr' then
    --     return 'IClrMult'
    elseif short == 'XClr' then
        return 'XClrMult'
    elseif short == 'IHlr' then
        return 'IHlrMult'
    elseif short == 'XHlr' then
        return 'XHlrMult'
    elseif short == 'ls' then
        return 'layerSizes'
    elseif short == 'kfn' then
        return 'kFoldNum'
    end
end

function runWithParams(paramFactorialSet)

    -- TODO: These are set to constants right now
    local modality = 'C'

    prepare()
    trainAndEvaluateAutomatic(modality, p.numEpochs, p.evalInterval, paramFactorialSet)
    saveResults()
end

function saveResults()

    if not matio then
        matio = require 'matio'
    end
    local filenameWOExt = 'paramResults_'.. g.startStatsId .. '_' .. g.endStatsId
    matio.save(g.statsDir .. '/' .. filenameWOExt .. '.mat', { trainResMat = g.trainResultsMatrix, valResMat = g.valResultsMatrix} )

    local results = {}
    results.legend = g.paramSettingsLegend
    results.trainResultsMatrix = g.trainResultsMatrix
    results.valResultsMatrix = g.valResultsMatrix
    results.evalInterval = p.evalInterval
    results.numEpochs = p.numEpochs
    torch.save(g.statsDir .. '/' .. filenameWOExt .. '.t7', results)
end

function prepare()

    -- TODO: These are set to constants right now
    local simWeight = 1

    if p.lrMultForHashLayer and not p.XHlrMult then
        p.XHlrMult = p.lrMultForHashLayer
        p.IHlrMult = p.lrMultForHashLayer
    end

    local XClrMult = p.XClrMult or 1
    local IClrMult = p.IClrMult or 1

    clearState()
    resetGlobals()
    loadFullModel(p.modelType, p.XHlrMult, p.IHlrMult, XClrMult, IClrMult, false, p.layerSizes)
    if d.trainset == nil then
        loadData()
    end
    if p.numKFoldSplits > 1 and d.kNumLoaded == nil or d.kNumLoaded ~= p.kFoldNum then
        loadTrainAndValSubsets(p.kFoldNum)
    end
    getOptimStateAndShareParameters('C')
    doGetCriterion(simWeight, p.balanceRegWeight, p.quantRegWeight)
end

function clearState()

    m = {}
    o = {}
    collectgarbage()
end

function getStatsFileName()

    local sDir = io.popen('dir \"' .. g.statsDir .. '/\"') 
    local listAsStr = sDir:read("*a") 
    io.close(sDir)

    local id = 1
    while string.match(listAsStr, "stats" .. id .. ".txt") do
        id = id + 1
    end

    -- Keep track of which stats files we have written so far
    if not g.startStatsId then
        g.startStatsId = id
    end
    g.endStatsId = id

    return "stats" .. id .. ".txt"
end

function printParams(paramFactorialSet, log1, log2)

    -- if not gc then
    --     gc = 38
    -- end
    -- print(gc)
    local fullStr = ''
    for i = 1, #paramFactorialSet do
        local shortParamName = paramFactorialSet[i][1]
        local longParamName = getLongParamName(shortParamName)
        local paramVal = p[longParamName]
        if type(paramVal) == 'string' then
            str = string.format("%s = %s", shortParamName, paramVal)
        elseif type(paramVal) == 'table' then
            str = '{ '
            for j = 1, #paramVal do
                str = str .. paramVal[j] .. ' '
            end
            str = str .. '}'
        else
            str = string.format("%s = %.5f", shortParamName, paramVal)
        end
        statsPrint(str, log1, log2)
        if shortParamName ~= 'kfn' then
            fullStr = fullStr .. str .. '\n' 
        end
    end
    print("\n")
    -- gc = gc + 1
    return fullStr
end

function getLegendSize()

    local idx = 1
    while(g.paramSettingsLegend[tostring(idx)]) do
        idx = idx + 1
    end
    return idx - 1
end

function trainAndEvaluateAutomatic(modality, numEpochs, evalInterval, paramFactorialSet)

  local resultsEvalIdx = 1

  local date = os.date("*t", os.time())
  local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min

  local statsFileName = getStatsFileName()
  g.plotFilename = g.statsDir .. '/plots/' .. statsFileName .. '_CMplot.pdf'
  g.sf = io.open(g.statsDir .. '/' .. statsFileName, 'w')
  g.meta:write(statsFileName .. '\n')
  print("Training with new parameters...")
  statsPrint(dateStr, g.meta, g.sf)
  local paramStr = printParams(paramFactorialSet, g.meta, g.sf)
  g.paramSettingsLegend[tostring(getLegendSize() + 1)] = paramStr
  g.snapshotFilename = statsFileName

  local count = 0
  local epoch = 0
  local bestEpochLoss = 1e10 -- a large number
  local bestCmEpochLoss = 1e10 -- best cross-modal epoch loss
  local bestIXv = 0
  local bestIXvEpoch = 0
  local bestAvgV = 0
  local bestAvgVEpoch = 0
  while epoch <= numEpochs and count < p.consecutiveStop do

    epoch = epoch + 1

    if epoch == 11 then
        changeLearningRateForHashLayer(1e4)
    elseif epoch == 51 then
        changeLearningRateForHashLayer(5e3)
    elseif epoch == 701 then
        changeLearningRateForHashLayer(1e3)
    end

    local epochLoss, cmEpochLoss = doOneEpochOnModality(modality, false)
    if cmEpochLoss < bestCmEpochLoss then
        bestCmEpochLoss = cmEpochLoss
    end
    if epochLoss < bestEpochLoss then
        bestEpochLoss = epochLoss
        -- count = 0
    -- else
        -- count = count + 1
    end

    local IXt, XIt, IXv, XIv
    if epoch % evalInterval == 0 then
      IXt, XIt, IXv, XIv = doRunEvals(g.resultsParamIdx, resultsEvalIdx)
      resultsEvalIdx = resultsEvalIdx + 1
      if IXv > bestIXv then
        bestIXv = IXv
        bestIXvEpoch = epoch
        if IXv > 85 then
            local name = g.snapshotFilename .. '_bestIXv'
            saveSnapshot(name, o.params_full, o.gradParams_full)
        end
      end
      local avgV = (IXv + XIv) / 2
      if avgV > bestAvgV then
        count = 0
        bestAvgV = avgV
        bestAvgVEpoch = epoch
        local name = g.snapshotFilename .. '_bestAvg'
        saveSnapshot(name, o.params_full, o.gradParams_full)
      else
        count = count + 1
      end
    end

    addPlotStats(epoch, evalInterval, IXt, XIt, IXv, XIv) -- IXt, XIt, etc. can be nil if this is not an eval epoch

    -- plotCrossModalLoss(epoch) -- TODO: This sometimes causes the program to crash. Plotting at end instead.
  end

  statsPrint('****Stopped at epoch ' .. epoch, g.meta, g.sf)
  statsPrint(string.format('Best epoch (avg) loss = %.2f', bestEpochLoss), g.meta, g.sf)
  statsPrint(string.format('Best cross-modal epoch (avg) loss = %.2f', bestCmEpochLoss), g.meta, g.sf)
  statsPrint(string.format('Best IXv = %.4f @ epoch %d\n\n', bestIXv, bestIXvEpoch), g.meta, g.sf)
  statsPrint(string.format('Best avgV = %.4f @ epoch %d\n\n', bestAvgV, bestAvgVEpoch), g.meta, g.sf)

  plotCrossModalLoss(epoch)
  gnuplot.closeall()

  io.close(g.sf)
end

function doRunEvals(paramIdx, evalIdx)

    local IXt, XIt, IXv, XIv = runEvals()
    local den = 2 * p.numKFoldSplits
    g.trainResultsMatrix[paramIdx][evalIdx] = g.trainResultsMatrix[paramIdx][evalIdx] + (IXt / den)
    g.trainResultsMatrix[paramIdx][evalIdx] = g.trainResultsMatrix[paramIdx][evalIdx] + (XIt / den)
    g.valResultsMatrix[paramIdx][evalIdx] = g.valResultsMatrix[paramIdx][evalIdx] + (IXv / den)
    g.valResultsMatrix[paramIdx][evalIdx] = g.valResultsMatrix[paramIdx][evalIdx] + (XIv / den)
    return IXt, XIt, IXv, XIv
end