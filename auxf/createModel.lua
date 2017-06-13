local nninit = require 'nninit'

function getHashLayerFullyConnected(prevLayerSize, hashLayerSize, lrMultForHashLayer, addHiddenLayer)

    local model = nn.Sequential()

    if addHiddenLayer then
        model:add(nn.Linear(prevLayerSize, prevLayerSize)
                 :init('weight', nninit.xavier, {dist = 'normal'})
                 :learningRate('weight', lrMultForHashLayer)
                 :learningRate('bias', lrMultForHashLayer))
        -- model:add(cudnn.ReLU(true))
        -- model:add(nn.Dropout(0.500000))
    end

    model:add(nn.Linear(prevLayerSize, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer)
        :learningRate('bias', lrMultForHashLayer))

    model:add(nn.Reshape(p.L, p.k))
    model:add(nn.SplitTable(2))

    map = nn.MapTable()
    map:add(nn.SoftMax())

    model:add(map)

    model:add(nn.JoinTable(2))

    return model
end

function getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, addHiddenLayer)

    local groupSize = math.ceil(prevLayerSize / L)

    local hashLayer = nn.Sequential()

    if addHiddenLayer then
        hashLayer:add(nn.Linear(prevLayerSize, groupSize * L)
                 :init('weight', nninit.xavier, {dist = 'normal'})
                 :learningRate('weight', lrMultForHashLayer)
                 :learningRate('bias', lrMultForHashLayer))
    end

    hashLayer:add(nn.Reshape(L, groupSize))
    hashLayer:add(nn.SplitTable(2))

    map1 = nn.MapTable()
    map1:add(nn.Linear(groupSize, k)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer)
        :learningRate('bias', lrMultForHashLayer))

    map2 = nn.MapTable()
    map2:add(nn.SoftMax())

    hashLayer:add(map1)
    hashLayer:add(map2)

    hashLayer:add(nn.JoinTable(2))

    return hashLayer
end

function getImageModelImageNetPretrained(lrMultForLastLayer, lrMultForClassLayer)
    -- Uses pre-trained model from https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

    -- TODO: lrMultForClassLayer is not implemented yet! Maybe don't need to implement it.

    local model = nn.Sequential()

    local imDir = '/home/kjoslyn/kevin/Project/IMAGENET/'
    local model = loadcaffe.load(imDir .. 'deploy.prototxt', imDir .. 'bvlc_alexnet.caffemodel', 'cudnn')
    model.modules[24] = nil
    model.modules[23] = nil
    model:add(nn.Linear(4096, p.numClasses)
            :init('weight', nninit.xavier, {dist = 'normal', gain = 'sigmoid'})
            :learningRate('weight', lrMultForLastLayer)
            :learningRate('bias', lrMultForLastLayer))

    model:add(nn.Sigmoid())

    model = model:cuda()

    return model
end

function getImageModel()

    -- local model = loadcaffe.load(g.filePath .. 'CNN Model/trainnet.prototxt', g.filePath .. 'CNN Model/snapshot_iter_16000.caffemodel', 'cudnn')
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3,96, 11, 11, 4, 4, 0, 0, 1):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(nn.View(-1):setNumInputDims(3))
    model:add(nn.Linear(9216, 4096):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(nn.Linear(4096, 4096):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(nn.Linear(4096, p.numClasses):init('weight', nninit.xavier, {dist = 'normal', gain = 'sigmoid'}))

    model:add(nn.Sigmoid())

    return model
end

function getImageModelForFullNet(L, k, type, lrMultForHashLayer)

    local model = getFineTunedImageModel()
    model.modules[#model.modules] = nil -- This is messy, but need to remove sigmoid layer for now. Will add it back later.
    return createClassifierAndHasher(model, 4096, L, k, type, lrMultForHashLayer)
end

function getFineTunedImageModel()

    local model = getImageModel()

    local snapshotFile 
    if p.datasetType == 'mir' then
        snapshotFile = 'id1.t7'
    elseif p.datasetType == 'nus' then
        snapshotFile = 'snapshot_epoch22.t7'
    end
    loadModelSnapshot(model, 'imageNet', snapshotFile)

    -- local snapshot2ndLevelDir = 'imageNet'
    -- local snapshotFile = 'snapshot_epoch_500.t7'
    -- loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFile)

    return model
end

function table.shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

function buildCustomTextModel(layerSizes, lrMultForClassLayer)

    local lrMult = lrMultForClassLayer or 1

    local model = nn.Sequential()

    local ls = table.shallow_copy(layerSizes)

    ls[#ls + 1] = p.numClasses

    -- { t, 2048 } is basic (2 hidden layers)
    -- p.tagDim -> p.tagDim (t)
    -- p.tagDim -> 2048     (2048)
    -- 2048 -> p.numClasses [Assumed]

    local lprev
    for i = 1, #ls do
        local from, to
        if i == 1 then
            from = p.tagDim
        else
            from = lprev 
        end

        local lt = ls[i]
        if lt == 't' then
            to = p.tagDim
        else
            to = lt
        end

        if i ~= 1 then
            model:add(cudnn.ReLU(true))
            model:add(nn.Dropout(0.500000))
        end

        local weightInit
        if not p.weightInit or p.weightInit == 'xavier' then
            weightInit = nninit.xavier
        elseif p.weightInit == 'kaiming' then
            weightInit = nninit.kaiming
        else
            print('Error: Unrecognized weight initialization scheme')
        end

        -- model:add(nn.Linear(from, to):init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'}))
        model:add(nn.Linear(from, to)
             :init('weight', weightInit, {dist = 'normal', gain = 'relu'})
             :learningRate('weight', lrMult)
             :learningRate('bias', lrMult))

        lprev = to
    end

    model:add(nn.Sigmoid())
    
    model = model:cuda()

    return model
end

function doGetBasicTextModel(lrMultForClassLayer)

    local lrMult = lrMultForClassLayer or 1

    local model = nn.Sequential()
    -- model.add(nn.View(-1):setNumInputDims(3))
    model:add(nn.Linear(p.tagDim, p.tagDim)
         :init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'})
         :learningRate('weight', lrMult)
         :learningRate('bias', lrMult))

    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))

    model:add(nn.Linear(p.tagDim, 2048)
         :init('weight', nninit.xavier, {dist = 'normal', gain = 'relu'})
         :learningRate('weight', lrMult)
         :learningRate('bias', lrMult))

    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))

    model:add(nn.Linear(2048, p.numClasses)
         :init('weight', nninit.xavier, {dist = 'normal', gain = 'sigmoid'})
         :learningRate('weight', lrMult)
         :learningRate('bias', lrMult))

    model:add(nn.Sigmoid())
    
    model = model:cuda()

    return model
end

function getUntrainedTextModel(layerSizes, lrMultForClassLayer)

    if not layerSizes then
        return doGetBasicTextModel(lrMultForClassLayer)
    else
        return buildCustomTextModel(layerSizes, lrMultForClassLayer)
    end
end

function getMirflickrCaffeTrainedTextModel()

    local model = loadcaffe.load(g.filePath .. 'text model/tag_trainnet.prototxt', g.filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer that comes from caffemodel
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    model:add(nn.Sigmoid())

    return model
end

local function checkTableEquivalence(tb1, tb2)
    if #tb1 ~= #tb2 then
        return false
    end
    for i = 1,#tb1 do
        if tb1[i] ~= tb2[i] then
            return false
        end
    end
    return true
end

function getTextModelForFullNet(L, k, type, lrMultForHashLayer, lrMultForClassLayer, layerSizes)

    local model = getUntrainedTextModel(layerSizes, lrMultForClassLayer)
    local snapshotFile 
    if p.datasetType == 'mir' then
        -- snapshotFile = '2hl_epoch250.t7'
        -- snapshotFile = 'sn1700.t7'
        -- If layerSizes parameter is not given, we will assume the standard case
        if not layerSizes or checkTableEquivalence(layerSizes, { 't', 2048 }) then
            snapshotFile = 'epoch330.t7'
        elseif checkTableEquivalence(layerSizes, { 2048, 2048, 2048 }) then
            snapshotFile = 'stats57.txt_best.t7'
        else
            print('Error in getTextModelForFullNet: Unrecognized model architecture')
        end
    elseif p.datasetType == 'nus' then
        -- snapshotFile = '2hl_epoch100.t7'
        snapshotFile = '1hl_epoch100.t7'
    end
    loadModelSnapshot(model, 'textNet', snapshotFile)

    -- if p.datasetType == 'mir' then
    --     model = getMirflickrCaffeTrainedTextModel()
    -- elseif p.datasetType == 'nus' then
    --     model = getUntrainedTextModel()
    --     local snapshot2ndLevelDir = 'textNet/Large'
    --     local snapshotFile = 'snapshot_epoch_500.t7'
    --     loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFile)
    -- end

    model.modules[#model.modules] = nil -- This is messy, but need to remove sigmoid layer for now. Will add it back later.

    return createClassifierAndHasher(model, 2048, L, k, type, lrMultForHashLayer)
end

function getHashLayer(prevLayerSize, type, L, k, lrMultForHashLayer)

    local hashLayer
    if type == 'hfc' then
        hashLayer = getHashLayerFullyConnected(prevLayerSize, L*k, lrMultForHashLayer, true) 
    elseif type == 'fc' then
        hashLayer = getHashLayerFullyConnected(prevLayerSize, L*k, lrMultForHashLayer, false) 
    elseif type == 'hgr' then
        hashLayer = getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, true)
    elseif type == 'gr' then
        hashLayer = getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, false)
    else
        print('ERROR: Unrecognized hash layer type')
    end

    return hashLayer
end

function createClassifierAndHasher(model, prevLayerSize, L, k, type, lrMultForHashLayer)

    -- Grab classification layer and remove it
    local classLayer = nn.Sequential()
    classLayer:add(model.modules[#model.modules])
    classLayer:add(nn.Sigmoid())
    model.modules[#model.modules] = nil

    local hashLayer = getHashLayer(prevLayerSize, type, L, k, lrMultForHashLayer)

    local concat = nn.ConcatTable()
    concat:add(classLayer)
    concat:add(hashLayer)

    model:add(concat)

    local classifier = nn.Sequential()
    classifier:add(model)
    classifier:add(nn.SelectTable(1))

    local hasher = nn.Sequential()
    hasher:add(model)
    hasher:add(nn.SelectTable(2))

    return classifier, hasher
end

function createCombinedModel(hasher1, hasher2)

    -- input to model is table of 4 tensors
    -- 1: Image input
    -- 2: Text input
    -- 3: beta_im
    -- 4: beta_te

    local model = nn.Sequential()

    -- First stage: Image and text hasher, forward 3 and 4

    local con1 = nn.ConcatTable()

    local con1h1 = nn.Sequential()
    con1h1:add(nn.SelectTable(1))
    con1h1:add(hasher1)

    local con1h2 = nn.Sequential()
    con1h2:add(nn.SelectTable(2))
    con1h2:add(hasher2)

    con1:add(con1h1)
    con1:add(con1h2)
    con1:add(nn.SelectTable(3))
    con1:add(nn.SelectTable(4))

    model:add(con1)

    -- Second Stage: Dot Product and regularizers

    local con2 = nn.ConcatTable()

    local con2dot = nn.Sequential()
    local con2dotSel = nn.ConcatTable()
    con2dotSel:add(nn.SelectTable(1))
    con2dotSel:add(nn.SelectTable(2))
    con2dot:add(con2dotSel)
    con2dot:add(nn.DotProduct())

    local bitBalancer1 = getBitBalancer(1, 3)
    local bitBalancer2 = getBitBalancer(2, 4)
    local quantizer1 = getQuantizer(1)
    local quantizer2 = getQuantizer(2)

    con2:add(con2dot)
    con2:add(bitBalancer1)
    con2:add(bitBalancer2)
    con2:add(quantizer1)
    con2:add(quantizer2)

    model:add(con2)

    model = model:cuda()

    return model
end

function getBitBalancer(inputIdx1, inputIdx2)

    local balancer = nn.Sequential()

    local balancerCon = nn.ConcatTable()
    balancerCon:add(nn.SelectTable(inputIdx1))
    balancerCon:add(nn.SelectTable(inputIdx2))

    balancer:add(balancerCon)
    balancer:add(nn.CMulTable())
    balancer:add(nn.Sum(2))

    return balancer
end

function getQuantizer(inputIdx)

    local quantizer = nn.Sequential()

    quantizer:add(nn.SelectTable(inputIdx))
    quantizer:add(nn.AddConstant(-.5))
    quantizer:add(nn.Abs())
    quantizer:add(nn.Sum(2))

    return quantizer
end

function getSiameseHasher(hasher)

    local model = nn.Sequential()

    -- local net2 = hasher:clone('weight', 'bias', 'gradWeight', 'gradBias') -- Don't do this and call share later
    local net2 = hasher:clone()

    local prl = nn.ParallelTable()
    prl:add(hasher)
    prl:add(net2)

    model:add(prl)
    model:add(nn.DotProduct())

    model = model:cuda()

    return model
end

function getCriterion(simWeight, balanceWeight, quantWeight)

    -- Cross-modal similarity criterion

    local criterion = nn.ParallelCriterion()

    local critSim = nn.MSECriterion()
    critSim.sizeAverage = false

    -- Bit balance criterions

    local critBalanceIm = nn.AbsCriterion()
    critBalanceIm.sizeAverage = false

    local critBalanceTe = nn.AbsCriterion()
    critBalanceTe.sizeAverage = false

    -- Quantization criterions

    local critQuantIm = nn.AbsCriterion()
    critQuantIm.sizeAverage = false

    local critQuantTe = nn.AbsCriterion()
    critQuantTe.sizeAverage = false

    -- Combined criterion

    criterion:add(critSim, simWeight)
    criterion:add(critBalanceIm, balanceWeight)
    criterion:add(critBalanceTe, balanceWeight)
    criterion:add(critQuantIm, quantWeight)
    criterion:add(critQuantTe, quantWeight)

    -- criterion = nn.MSECriterion()
    -- -- criterion = nn.BCECriterion()
    -- criterion.sizeAverage = false

    criterion = criterion:cuda()

    return criterion
end

function loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFileName)

  -- If these aren't specified, use hardcoded values
  if not snapshot2ndLevelDir and snapshotFileName then
    -- local snapshot2ndLevelDir = 'Lr5e4_5kquery_5kdatabase'
    local snapshot2ndLevelDir = 'imageNet'
    local snapshotFileName = 'snapshot_epoch_500.t7'
  end

  local snapshotFullPath
  if not snapshot2ndLevelDir then
    print('****Loading snapshot: ' .. snapshotFileName)
    snapshotFullPath = g.snapshotDir .. '/' .. snapshotFileName
  else
    print('****Loading snapshot: ' .. snapshot2ndLevelDir .. '/' .. snapshotFileName)
    snapshotFullPath = g.snapshotDir .. '/' .. snapshot2ndLevelDir .. '/' .. snapshotFileName
  end

  local snapshot = torch.load(snapshotFullPath)
  local N = snapshot.params:size(1)

  local params, gparams = model:getParameters()

  local batchSize = 1e5

  local numBatches = torch.ceil(N / batchSize)
  for b = 0, numBatches - 1 do
      local startIndex = b * batchSize + 1
      local endIndex = math.min((b + 1) * batchSize, N)
      params[ {{ startIndex, endIndex }} ]:copy(snapshot.params[ {{ startIndex, endIndex }} ])
  end

end

