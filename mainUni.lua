
require 'unimodal'

function getTestPFS()
    rap_pfs = torch.load('/home/kjoslyn/kevin/Project/autoStats/mirflickr/testPFS.t7')
    rap = rap_pfs.rap
    pfs = rap_pfs.pfs
end

function runAllParamsets(datasetType, modality, paramFactorialSet, numEpochs, plotNumEpochs, consecutiveStop)

    -- This is the main function to call

    loadParamsAndPackages(datasetType, modality, plotNumEpochs)

    -- g.skipPlot = true -- Don't plot the class accuracy like usual

    local autoStatsDir
    if datasetType == 'mir' then
        autoStatsDir = 'mirflickr'
    elseif datasetType == 'nus' then
        autoStatsDir = 'nuswide'
    end
    if modality == 'X' then
        autoStatsDir = autoStatsDir .. '/textNet'
    elseif modality == 'I' then
        autoStatsDir = autoStatsDir .. '/imageNet'
    end

    g.statsDir = '/home/kjoslyn/kevin/Project/autoStats/' .. autoStatsDir
    g.meta = io.open(g.statsDir .. "/metaStats.txt", 'a')
    g.startStatsId = nil

    p.numEpochs = numEpochs
    -- TODO: consecutiveStop is not implemented!
    if consecutiveStop then
        p.consecutiveStop = consecutiveStop
    else
        p.consecutiveStop = numEpochs + 1
    end

    local numParamCombs = getNumParamCombs(paramFactorialSet)
    p.numKFoldSplits = getNumKFoldSplits(paramFactorialSet)

    g.paramSettingsLegend = {}
    g.valMaxMatrix = torch.Tensor(numParamCombs, numEpochs / g.plotNumEpochs):fill(0) -- g.plotNumEpochs comes from loadParamsAndPackages
    g.avgLossMatrix = torch.Tensor(numParamCombs, numEpochs / g.plotNumEpochs):fill(0) -- g.plotNumEpochs comes from loadParamsAndPackages
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
    return true
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
        return 'baseLearningRate'
    elseif short == 'ls' then
        return 'layerSizes'
    elseif short == 'lrd' then
        return 'baseLearningRateDecay'
    elseif short == 'wd' then
        return 'baseWeightDecay'
    elseif short == 'mom' then
        return 'baseMomentum'
    elseif short == 'bs' then -- This is set to 100 by default in loadParamsAndPackages
        return 'batchSize'
    elseif short == 'wi' then
        return 'weightInit'
    elseif short == 'kfn' then
        return 'kFoldNum'
    end
end

function runWithParams(paramFactorialSet)

    prepare()
    trainAndEvaluateAutomatic(paramFactorialSet)
    saveResults()
end

function saveResults()

    if not matio then
        matio = require 'matio'
    end
    local filenameWOExt = 'paramResults_'.. g.startStatsId .. '_' .. g.endStatsId
    matio.save(g.statsDir .. '/' .. filenameWOExt .. '.mat', { valMaxMatrix = g.valMaxMatrix, avgLossMatrix = g.avgLossMatrix} )

    local results = {}
    results.legend = g.paramSettingsLegend
    results.valMaxMatrix = g.valMaxMatrix
    results.avgLossMatrix = g.avgLossMatrix
    results.plotNumEpochs = g.plotNumEpochs
    results.numEpochs = p.numEpochs
    torch.save(g.statsDir .. '/' .. filenameWOExt .. '.t7', results)
end

function prepare()

    clearState()
    resetPlotStats()
    loadModel() -- uses p.layerSizes to build model

    if d.trainset == nil then
        loadData()
    end
end

function clearState()

    m = {}
    o = {} -- This is actually not used now in unimodal.lua
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

function trainAndEvaluateAutomatic(paramFactorialSet)

  local date = os.date("*t", os.time())
  local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min

  local statsFileName = getStatsFileName()
  g.sf = io.open(g.statsDir .. '/' .. statsFileName, 'w')
  g.meta:write(statsFileName .. '\n')
  print("Training with new parameters...")
  statsPrint(dateStr, g.meta, g.sf)
  local paramStr = printParams(paramFactorialSet, g.meta, g.sf)
  g.paramSettingsLegend[tostring(getLegendSize() + 1)] = paramStr

  g.numEpochsCompleted = 0
  g.plotFilename = statsFileName .. '_plot.pdf'
  trainAndEvaluate(p.numEpochs, p.batchSize)

  g.valMaxMatrix[g.resultsParamIdx] = g.maxDataAcc
  g.avgLossMatrix[g.resultsParamIdx] = g.avgDataLoss

  statsPrint(string.format('***** Finished run'), g.meta, g.sf)
  statsPrint(string.format('Best val accuracy = %.2f', g.maxDataAcc:max()), g.meta, g.sf)
  statsPrint(string.format('Best training avg loss (over 5 epochs) = %.2f\n\n', g.avgDataLoss:min()/100), g.meta, g.sf)

  io.close(g.sf)
end