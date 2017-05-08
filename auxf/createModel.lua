local nninit = require 'nninit'

function getHashLayerFullyConnected(prevLayerSize, hashLayerSize, lrMultForHashLayer, addHiddenLayer)

    local model = nn.Sequential()

    if addHiddenLayer then
        model:add(nn.Linear(prevLayerSize, prevLayerSize)
                 :init('weight', nninit.xavier, {dist = 'normal'})
                 :learningRate('weight', lrMultForHashLayer))
        -- model:add(cudnn.ReLU(true))
        -- model:add(nn.Dropout(0.500000))
    end

    model:add(nn.Linear(prevLayerSize, hashLayerSize)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    model:add(nn.Reshape(p.L, p.k))
    model:add(nn.SplitTable(2))

    map = nn.MapTable()
    map:add(nn.SoftMax())

    model:add(map)

    model:add(nn.JoinTable(2))

    return model
end

function getHashLayerGrouped(prevLayerSize, L, k, lrMultForHashLayer, addHiddenLayer)

    local groupSize = prevLayerSize / L

    local hashLayer = nn.Sequential()

    if addHiddenLayer then
        hashLayer:add(nn.Linear(prevLayerSize, prevLayerSize)
                 :init('weight', nninit.xavier, {dist = 'normal'})
                 :learningRate('weight', lrMultForHashLayer))
    end

    hashLayer:add(nn.Reshape(L, groupSize))
    hashLayer:add(nn.SplitTable(2))

    map1 = nn.MapTable()
    map1:add(nn.Linear(groupSize, k)
        :init('weight', nninit.xavier, {dist = 'normal'})
        :learningRate('weight', lrMultForHashLayer))

    map2 = nn.MapTable()
    map2:add(nn.SoftMax())

    hashLayer:add(map1)
    hashLayer:add(map2)

    hashLayer:add(nn.JoinTable(2))

    return hashLayer
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
    model:add(nn.Linear(4096, 24):init('weight', nninit.xavier, {dist = 'normal', gain = 'sigmoid'}))

    model:add(nn.Sigmoid())

    return model
end

function getImageModelForFullNet(L, k, type, lrMultForHashLayer)

    local model = getImageModel()

    local snapshot2ndLevelDir = 'imageNet'
    local snapshotFile = 'snapshot_epoch_500.t7'
    loadModelSnapshot(model, snapshot2ndLevelDir, snapshotFile)

    model.modules[#model.modules] = nil -- This is messy, but need to remove sigmoid layer for now. Will add it back later.
    return createClassifierAndHasher(model, 4096, L, k, type, lrMultForHashLayer)
end

function getTextModelForFullNet(L, k, type, lrMultForHashLayer)

    local model = loadcaffe.load(g.filePath .. 'text model/tag_trainnet.prototxt', g.filePath .. 'text model/snapshot_iter_200.caffemodel', 'cudnn')

    -- Remove first layer that comes from caffemodel
    model.modules[1] = nil
    for i = 1,#model.modules-1 do
        model.modules[i] = model.modules[i+1]
    end
    model.modules[#model.modules] = nil

    return createClassifierAndHasher(model, 2048, L, k, type, lrMultForHashLayer)
end

function getTextModel2()
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

function getImageModel2()

    local model = loadcaffe.load(g.filePath .. 'CNN Model/trainnet.prototxt', g.filePath .. 'CNN Model/snapshot_iter_16000.caffemodel', 'cudnn')
    model:add(nn.Sigmoid())
    return model
end


function createClassifierAndHasher(model, prevLayerSize, L, k, type, lrMultForHashLayer)

    -- Grab classification layer and remove it
    local classLayer = nn.Sequential()
    classLayer:add(model.modules[#model.modules])
    classLayer:add(nn.Sigmoid())
    model.modules[#model.modules] = nil

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

  print('****Loading snapshot: ' .. snapshot2ndLevelDir .. '/' .. snapshotFileName)

  local snapshotFullPath = g.snapshotDir .. '/' .. snapshot2ndLevelDir .. '/' .. snapshotFileName
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

