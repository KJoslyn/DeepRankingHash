require 'fullNet'

function runAllParamsets(datasetType, paramFactorialSet, numEpochs, evalInterval, iterationsPerEpoch, usePretrainedImageFeatures, consecutiveStop, startStatsId)

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

    g.statsDir = g.userPath .. '/kevin/Project/autoStats/' .. autoStatsDir
    g.meta = io.open(g.statsDir .. "/metaStats.txt", 'a')
    g.startStatsId = startStatsId

    p.numEpochs = numEpochs
    p.evalInterval = evalInterval
    if consecutiveStop then
        p.consecutiveStop = consecutiveStop
    else
        p.consecutiveStop = numEpochs + 1
    end

    local numParamCombs = getNumParamCombs(paramFactorialSet)
    p.numRuns = getNumRuns(paramFactorialSet)

    g.paramSettingsLegend = {}
    -- TODO: Convert to concatenating tensors (due to validating params)
    g.trainResultsMatrix = torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
    g.valResultsMatrix = torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
    g.resultsParamIdx = 0

    recursiveRunAllParamsets(paramFactorialSet, paramFactorialSet, 0, #paramFactorialSet)
    -- recursiveCreatePfsS(paramFactorialSet, paramFactorialSet, 0, #paramFactorialSet)

    io.close(g.meta)
end

function fixPfsS(pfsS)

-- Assumptions: 'wd' is the 3rd parameter in pfs
-- 'lrd' is the 4th parameter in pfs
-- Actually, the two can be reversed

    for i = 1,#pfsS do
        local pfs = pfsS[i]
        pfs[1] = pfs[2]
        pfs[2] = pfs[4]
        pfs[5] = {'wd', {0}}
        pfs[6] = {'lrd', {0}}
        for j = 5,12 do
            pfs[j-2] = pfs[j]
        end
        pfs[11] = nil
        pfs[12] = nil
    end
end

local function recursiveCreatePfsS(pfs_part, pfs_full, paramCount, numParams)

    if paramCount == numParams then
        -- printParams(pfs_full)
        if validateParams() then
            if not g.pfsS then g.pfsS = {} end
            local pfsS_1 = table.shallow_copy(g.pfsS_one)
            local i = 1
            for key, value in pairs(pfsS_1) do
                pfsS_1[i] = {key,{value}}
                pfsS_1[key] = nil
                i = i + 1
            end
            g.pfsS[#g.pfsS + 1] = pfsS_1
        end
    else
        local thisParamset = pfs_part[1]
        local paramName = thisParamset[1]
        local valueSet = thisParamset[2]
        for _, value in pairs(valueSet) do

            local oldValue = setParamValue(paramName, value)
            if not g.pfsS_one then g.pfsS_one = {} end
            g.pfsS_one[paramName] = value
            recursiveCreatePfsS( { unpack(pfs_part, 2, #pfs_part) }, pfs_full, paramCount + 1, numParams)
            setParamValue(paramName, oldValue)
        end
    end
end

function createPfsS(pfs)

    require 'auxf.createModel'
    p = {}
    g = {}
    local userPath = os.getenv("HOME") -- will be either '/home/kejosl' or '/home/kjoslyn'

    recursiveCreatePfsS(pfs, pfs, 0, #pfs)

    fixPfsS(g.pfsS)
    torch.save(userPath .. '/kevin/Project/temp/pfsS.t7', g.pfsS)
end

function recursiveRunAllParamsets(pfs_part, pfs_full, paramCount, numParams)

    if paramCount == numParams then
        -- printParams(pfs_full)
        if validateParams() then
            -- If this starts a new parameter combination, increment the paramIdx (initialized to 0)
            if p.numRuns == 1 or p.run == 1 then
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
    elseif p.balanceRegWeight == 0 and p.quantRegWeight == 0 then
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
        if pfs[i][1] ~= 'run' then
            count = count * #pfs[i][2]
        end
    end
    return count
end

function getNumRuns(pfs)

    for i = 1,#pfs do
        if pfs[i][1] == 'run' then
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
    elseif short == 'run' then
        return 'run'
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
    getOptimStateAndShareParameters('C')
    doGetCriterion(simWeight, p.balanceRegWeight, p.quantRegWeight)
end

function clearState()

    m = {}
    o = {}
    collectgarbage()
end

function getStatsFileName()

    -- -- We do not look at the directory contents every time simply because an error was raised on kejosl
    -- -- upon the second time of listing the directory contents -- TODO: Figure out why
    -- if not g.startStatsId then
    --     local sDir = io.popen('dir \"' .. g.statsDir .. '/\"') 
    --     local listAsStr = sDir:read("*a") 
    --     io.close(sDir)

    --     local id = 1
    --     while string.match(listAsStr, "stats" .. id .. ".txt") do
    --         id = id + 1
    --     end

    --     -- Keep track of which stats files we have written so far
    --     g.startStatsId = id
    --     g.endStatsId = id
    -- else
    --     g.endStatsId = g.endStatsId + 1
    -- end

    -- return "stats1.txt"

    -- -- endStatsId is also essentially the 'current' stats id
    if g.endStatsId then
        g.endStatsId = g.endStatsId + 1
    else
        g.endStatsId = g.startStatsId
    end
    return "stats" .. g.endStatsId .. ".txt"
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
        fullStr = fullStr .. str .. '\n'
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
    elseif epoch == 401 then -- 701
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
    local suffix1 = '_bestIXv'
    local suffix2 = '_bestAvg'
    if epoch % evalInterval == 0 then
      IXt, XIt, IXv, XIv = doRunEvals(g.resultsParamIdx, resultsEvalIdx)
      resultsEvalIdx = resultsEvalIdx + 1
      if IXv > bestIXv then
        bestIXv = IXv
        bestIXvEpoch = epoch
        -- local suffix = '_bestIXv'
        saveSnapshot(g.snapshotFilename .. suffix1, o.params_full, o.gradParams_full)
        prepareValAndTestMAPs(suffix1)
      end
      local avgV = (IXv + XIv) / 2
      if avgV > bestAvgV then
        count = 0
        bestAvgV = avgV
        bestAvgVEpoch = epoch
        -- local suffix = '_bestAvg'
        saveSnapshot(g.snapshotFilename .. suffix2, o.params_full, o.gradParams_full)
        prepareValAndTestMAPs(suffix2)
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

--   local g_s = {}
--   g_s.g = g
--   g_s.s = s
--   torch.save(g.snapshotDir .. '/stats' .. g.endStatsId .. '_g_s.t7', g_s)
  -- There is a problem sometimes with plotting on kejosl which causes an error
  if g.userPath == '/home/kjoslyn' then
    plotCrossModalLoss(epoch)
    gnuplot.closeall()
  end

  io.close(g.sf)
end

-- TODO: Make these test and not val
function prepareValAndTestMAPs(suffix)

    local classesTo
    -- if p.datasetType == 'mir' then
        classesTo = {'training','val','pretraining'}
    -- else
    --     classesTo = {'training','val'}
    -- end

    local ixv_name = g.snapshotFilename .. '_DS_data_IX_val' .. suffix .. '.mat'
    local IXv = calcMAP(I, X, 'val', classesTo, true, ixv_name)

    local xiv_name = g.snapshotFilename .. '_DS_data_XI_val' .. suffix .. '.mat'
    local XIv = calcMAP(X, I, 'val', classesTo, true, xiv_name)

    local ixt_name = g.snapshotFilename .. '_DS_data_IX_test' .. suffix .. '.mat'
    local IXt = calcMAP(I, X, 'query', classesTo, true, ixt_name)

    local xit_name = g.snapshotFilename .. '_DS_data_XI_test' .. suffix .. '.mat'
    local XIt = calcMAP(X, I, 'query', classesTo, true, xit_name)
end

function doRunEvals(paramIdx, evalIdx)

    local IXt, XIt, IXv, XIv = runEvals()
    local den = 2 * p.numRuns
    g.trainResultsMatrix[paramIdx][evalIdx] = g.trainResultsMatrix[paramIdx][evalIdx] + (IXt / den)
    g.trainResultsMatrix[paramIdx][evalIdx] = g.trainResultsMatrix[paramIdx][evalIdx] + (XIt / den)
    g.valResultsMatrix[paramIdx][evalIdx] = g.valResultsMatrix[paramIdx][evalIdx] + (IXv / den)
    g.valResultsMatrix[paramIdx][evalIdx] = g.valResultsMatrix[paramIdx][evalIdx] + (XIv / den)
    return IXt, XIt, IXv, XIv
end