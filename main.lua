require 'fullNet'

function runAllParamsets(paramFactorialSet)

    -- This is the main function to call

    -- TODO: This is set to a constants
    local iterationsPerEpoch = 25

    loadParamsAndPackages(iterationsPerEpoch)

    g.statsDir = '/home/kjoslyn/kevin/Project/autoStats'
    g.meta = io.open(g.statsDir .. "/metaStats.txt", 'a')

    recursiveRunAllParamsets(paramFactorialSet, paramFactorialSet, 0, #paramFactorialSet)

    io.close(g.meta)
end

function recursiveRunAllParamsets(pfs_part, pfs_full, paramCount, numParams)

    if paramCount == numParams then
        runWithParams(pfs_full)
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
    elseif short == 'bw' then
        return 'balanceRegWeight'
    elseif short == 'qw' then
        return 'quantRegWeight'
    end
end

function runWithParams(paramFactorialSet)

    -- TODO: These are set to constants right now
    local modality = 'C'
    -- local numEpochs = 100
    local numEpochs = 1
    local evalInterval = 5

    prepare()
    trainAndEvaluateAutomatic(modality, numEpochs, evalInterval, paramFactorialSet)
end

function prepare()

    -- TODO: These are set to constants right now
    local kNum = 1
    local modelType = 'hfc'
    local simWeight = 1

    clearState()
    loadFullModel(modelType, p.lrMultForHashLayer)
    if d.trainset == nil then
        loadData()
    end
    if d.kNumLoaded == nil or d.kNumLoaded ~= kNum then
        loadTrainAndValSubsets(kNum)
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
    return "stats" .. id .. ".txt"
end

function printParams(paramFactorialSet, log1, log2)

    for i = 1, #paramFactorialSet do
        local shortParamName = paramFactorialSet[i][1]
        local longParamName = getLongParamName(shortParamName)
        local paramVal = p[longParamName]
        statsPrint(string.format("%s = %.2f", shortParamName, paramVal), log1, log2)
    end
end

function trainAndEvaluateAutomatic(modality, numEpochs, evalInterval, paramFactorialSet)

  local date = os.date("*t", os.time())
  local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min

  local statsFileName = getStatsFileName()
  g.sf = io.open(g.statsDir .. '/' .. statsFileName, 'w')
  g.meta:write(statsFileName .. '\n')
  print("Training with new parameters...")
  statsPrint(dateStr, g.meta, g.sf)
  printParams(paramFactorialSet, g.meta, g.sf)

  local count = 0
  local epoch = 1
  local bestEpochLoss = 1e10 -- a large number
  while epoch <= numEpochs and count < 10 do

    local epochLoss = doOneEpochOnModality(modality, epoch, false)
    if epochLoss < bestEpochLoss then
        bestEpochLoss = epochLoss
        count = 0
    else
        count = count + 1
    end

    if epoch % evalInterval == 0 then
      runEvals()
    end
    epoch = epoch + 1
  end

  statsPrint('****Stopped at epoch ' .. epoch, g.meta, g.sf)
  statsPrint(string.format('Best epoch (avg) loss = %.2f\n\n', bestEpochLoss), g.meta, g.sf)

  io.close(g.sf)
end